import torch
import nltk
import numpy as np
import random
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration

vocab_size_dic = {'meta-llama/Llama-3.1-8B': 128256,
                  'facebook/opt-13b': 50272,
                  'facebook/opt-1.3b': 50272 }
                  
def get_score_inverse(text, prompt, tokenizer, vocab_size, key_func, Y_func, key, c, seeding_scheme, input_type='text'):
    """Get pivotal value from a text.
       The input of prompt should be tokens.
       return: a numpy array
       Notice that in this process we may loss several watermark signal due to the skip of specail tokens.
    """

    Y_scores = []
    generator = torch.Generator()
    if input_type == 'text':
        encoded_text = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)[0]
        tokens = torch.cat((prompt, encoded_text), dim=0)
    elif input_type == 'token':
        tokens = torch.cat((prompt, text), dim=0)
    else:
        raise ValueError("Unsupported input type")
    for i in range(len(prompt), len(tokens)):
        inputs = tokens[:i].unsqueeze(0)
        token = tokens[i].unsqueeze(0).unsqueeze(1)
        xi, pi = key_func(generator,inputs, vocab_size, key, c, seeding_scheme)
        Y, U, eta = Y_func(token, pi, xi)
        Y_scores.append(Y.squeeze().numpy())
    return np.array(Y_scores)

def get_score_gumbel(text, prompt, tokenizer, vocab_size, key_func, Y_func, key, c, seeding_scheme, input_type='text'):
    """Get pivotal value from a text.
       The input of prompt should be tokens.
       return: a numpy array
       Notice that in this process we may loss several watermark signal due to the skip of specail tokens.
    """

    Y_scores = []
    generator = torch.Generator()
    if input_type == 'text':
        encoded_text = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)[0]
        tokens = torch.cat((prompt, encoded_text), dim=0)
    elif input_type == 'token':
        tokens = torch.cat((prompt, text), dim=0)
    else:
        raise ValueError("Unsupported input type")
    for i in range(len(prompt), len(tokens)):
        inputs = tokens[:i].unsqueeze(0)
        token = tokens[i].unsqueeze(0).unsqueeze(1)
        xi, pi = key_func(generator,inputs, vocab_size, key, c, seeding_scheme)
        Y = Y_func(token, pi, xi)
        Y_scores.append(Y.numpy())
    return np.array(Y_scores)

def replace_random(data, k, alg):
    """
    Replace the top k% highest values in each row of the array with random numbers.

    Parameters:
        data (numpy.ndarray): A 2D array of shape (n, m).
        k (float): Percentage of top values to replace (e.g., 5 for 5%).

    Returns:
        numpy.ndarray: The modified array.
    """
    # Validate inputs
    if not (0 < k <= 100):
        raise ValueError("k must be a percentage value between 0 and 100.")

    data = np.array(data)  # Ensure input is a NumPy array
    n_rows, n_cols = data.shape
    num_to_replace = int(np.ceil(k / 100 * n_cols))  # Number of elements to replace per row

    modified_data = data.copy()
    if alg == 'EXP':
        generate_samples = lambda size: np.random.uniform(0, 1, size)
    elif alg == 'transform':
        generate_samples = lambda size: -(1 - np.sqrt(1 - np.random.uniform(0, 1, size)))
    for i in range(n_rows):
        # Get the indices of the top k% values in the row
        top_k_indices = np.argsort(modified_data[i])[-num_to_replace:]
        # Replace these values with random numbers
        modified_data[i, top_k_indices] = generate_samples(num_to_replace)

    return modified_data

class DipperParaphraser():
    """Paraphrase a text using the DIPPER model."""

    def __init__(self, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration, device='cuda',
                 lex_diversity: int = 60, order_diversity: int = 0, sent_interval: int = 1, **kwargs):
        """
            Paraphrase a text using the DIPPER model.

            Parameters:
                tokenizer (T5Tokenizer): The tokenizer for the DIPPER model.
                model (T5ForConditionalGeneration): The DIPPER model.
                device (str): The device to use for inference.
                lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                sent_interval (int): The number of sentences to process at a time.
        """
        self.tokenizer = tokenizer
        self.model = model.eval().to(device)
        self.device = device
        self.lex_diversity = lex_diversity
        self.order_diversity = order_diversity
        self.sent_interval = sent_interval
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)

        # Validate diversity settings
        self._validate_diversity(self.lex_diversity, "Lexical")
        self._validate_diversity(self.order_diversity, "Order")
    
    def _validate_diversity(self, value: int, type_name: str):
        """Validate the diversity value."""
        if value not in [0, 20, 40, 60, 80, 100]:
            raise DiversityValueError(type_name)

    def edit(self, text: str, reference: str):
        """Edit the text using the DIPPER model."""
        # Calculate the lexical and order diversity codes
        lex_code = int(100 - self.lex_diversity)
        order_code = int(100 - self.order_diversity)

        # Preprocess the input text
        text = " ".join(text.split())
        sentences = sent_tokenize(text)
        
        # Preprocess the reference text
        prefix = " ".join(reference.replace("\n", " ").split())
        
        output_text = ""
        
        # Process the input text in sentence windows
        for sent_idx in range(0, len(sentences), self.sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + self.sent_interval])
            # Prepare the input for the model
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"
            # Tokenize the input
            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            if len(final_input["input_ids"][0]) > 512:
                break
            final_input = {k: v.to(self.device) for k, v in final_input.items()}
            # Generate the edited text
            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **self.gen_kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Update the prefix and output text
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


class WordDeletion():
    """Delete words randomly from the text."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the word deletion editor.

            Parameters:
                ratio (float): The ratio of words to delete.
        """
        self.ratio = ratio

    def edit(self, text: str, reference=None):
        """Delete words randomly from the text."""

        # Handle empty string input
        if not text:  
            return text

        # Split the text into words and randomly delete each word based on the ratio
        word_list = text.split()
        edited_words = [word for word in word_list if random.random() >= self.ratio]

        # Join the words back into a single string
        deleted_text = ' '.join(edited_words)

        return deleted_text


class SynonymSubstitution():
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the synonym substitution editor.

            Parameters:
                ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        nltk.download('wordnet')

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet."""
        words = text.split()
        num_words = len(words)
        
        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)
        
            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text

        