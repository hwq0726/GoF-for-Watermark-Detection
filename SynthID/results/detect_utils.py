import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import chi2, gamma


class ConfigManager:
    def __init__(self):
        # Initial configurations for each statistical test
        self.configs = {
            "z_score": {
                "gamma": 0.5,
                "alpha": 0.05,
                "print_out": True
            },
            "head_runs": {
                "skip": 49,
                "alpha": 0.05,
                "simulation_size": 1000,
                "print_out": True
            },
            "phi_divergence": {
                "alpha": 0.05,
                "s": 2,
                "mask": True,
                "print_out": True
            },
            "ars": {
                "ratio": 1,
                "alpha": 0.05,
                "print_out": True
            },
            "log": {
                "ratio": 1,
                "alpha": 0.05,
                "print_out": True
            },
            "sum_based": {
                "delta0": 0.2,
                "alpha": 0.05,
                "print_out": True
            },
            "sum_based_its": {
                "delta": 0.1,
                "weight": 1e-6,
                "vocab_size": 50265,
                "alpha": 0.05,
                "model_name": "1p3B",
                "print_out": True
            },
            "kolmogorov": {
                "alpha": 0.05,
                "print_out": True
            },
            "kuiper": {
                "alpha": 0.05,
                "skip": 1,  # skip the first score
                "print_out": True
            },
            "anderson": {
                "alpha": 0.05,
                "print_out": True
            },
            "cramer": {
                "alpha": 0.05,
                "print_out": True
            },
            "watson": {
                "alpha": 0.05,
                "skip": 1,  # skip the first score
                "print_out": True
            },
            "neyman": {
                "alpha": 0.05,
                "k": 3,
                "print_out": True
            },
            "chi_squared": {
                "alpha": 0.05,
                "c": 20,    # number of cells
                "skip": 20,    # start length
                "print_out": True
            },
            "rao": {
                "alpha": 0.05,
                "print_out": True
            },
            "greenwood": {
                "alpha": 0.05,
                "print_out": True
            },
            "its_neg": {
                "alpha": 0.05,
                "print_out": True
            },
            "simple_sum": {
                "alpha": 0.05,
                "print_out": True
            }
        }

    def set_config(self, test_name, **kwargs):
        """Update the configuration for a specific test."""
        if test_name in self.configs:
            self.configs[test_name].update(kwargs)
        else:
            raise ValueError(f"Unsupported test name: {test_name}")

    def get_config(self, test_name):
        """Retrieve the configuration for a specific test."""
        if test_name in self.configs:
            return self.configs[test_name]
        else:
            raise ValueError(f"Unsupported test name: {test_name}")

    def display_config(self, test_name=None):
        """Display the configuration for a specific test or all configurations."""
        if test_name:
            if test_name in self.configs:
                print(f"Configuration for {test_name}: {self.configs[test_name]}")
            else:
                raise ValueError(f"Unsupported test name: {test_name}")
        else:
            print("All configurations:")
            for key, value in self.configs.items():
                print(f"{key}: {value}")


class PhiDivergenceTest:
    def __init__(self, alpha=0.05, s=2, mask=True):
        self.alpha = alpha
        self.s = s
        self.mask = mask

    def compute_quantile(self, m):
        raw_data = np.random.uniform(size=(10000, m))
        H0s = self.compute_score(raw_data)
        log_H0s = np.log(H0s + 1e-10)
        q = np.quantile(log_H0s, 1 - self.alpha)
        return q

    def compute_score(self, Ys, a=1):
        ps = 1 - Ys
        ps = np.sort(ps, axis=-1)
        m = ps.shape[-1]
        first = int(m * a)
        ps = ps[..., :first]
        rk = np.arange(1, 1 + first) / first
        final = self.score_calculator(ps, rk, m)
        if self.mask:
            ind = (ps >= 1e-3) * (rk >= ps)
            final *= ind
        return m * np.max(final, axis=-1)

    def score_calculator(self, ps, rk, m):
        s = self.s
        eps = 1e-10
        if s == 1:
            final = rk * np.log(rk + eps) - rk * np.log(ps + eps) + (1 - rk + eps) * np.log(1 - rk + eps) - (
                        1 - rk) * np.log(1 - ps + eps)
        elif s == 0:
            final = ps * np.log(ps + eps) - ps * np.log(rk + eps) + (1 - ps + eps) * np.log(1 - ps + eps) - (
                        1 - ps) * np.log(1 - rk + eps)
        elif s == 2:
            final = (rk - ps) ** 2 / (ps * (1 - ps) + eps) / 2
        elif s == 1 / 2:
            final = 2 * (np.sqrt(rk) - np.sqrt(ps)) ** 2 + 2 * (np.sqrt(1 - rk) - np.sqrt(1 - ps)) ** 2
        elif s >= 0:
            final = (1 - (rk ** s) * (ps + eps) ** (1 - s) - ((1 - rk) ** s) * ((1 - ps + eps) ** (1 - s))) / (
                        s * (1 - s))
        elif s == -1:
            final = (rk - ps) ** 2 / (rk * (1 - rk) + eps) / 2
        else:  # -1 < s < 0
            final = (1 - ps ** (1 - s) / (rk + eps) ** (-s) - (1 - ps) ** (1 - s) / (1 - rk + eps) ** (-s)) / (
                        s * (1 - s))
        return final

    def HC_for_a_given_fraction(self, Ys, ratio):
        m = Ys.shape[-1]
        given_m = int(ratio * m) if ratio <= 1 and isinstance(ratio, float) else ratio
        truncated_Ys = Ys[..., 1:1+given_m]
        HC = self.compute_score(truncated_Ys)
        log_critical_value = self.compute_quantile(given_m)
        return HC, log_critical_value

    def analyze_y(self, Y):
        used_m = Y.shape[-1]
        x = np.arange(1, 1 + used_m, 10)
        y = []
        stats_from_data = []
        for x_point in x:
            HC, log_critical_value = self.HC_for_a_given_fraction(Y, x_point)
            stats = np.log(HC + 1e-10)
            stats_from_data.append(stats)
            mean = np.mean(stats >= log_critical_value)
            y.append(mean)
        stats_from_data = np.array(stats_from_data).T
        return np.array(y), stats_from_data


class StatisticalTests:
    # Default styles and colors as class attributes
    default_line_styles = [":", "-.", "-", "--"]
    default_colors = [
        '#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#17becf', '#bcbd22', '#17a2b8'
    ]

    def __init__(self, data, config_manager, algorithm, simulation_size=10000):
        self.data = data
        self.text_len = data.shape[1]
        self.config_manager = config_manager
        self.alg = algorithm
        self.simulation_size = simulation_size
        self.phi_divergence = PhiDivergenceTest()
        self.test_methods = {
            'kolmogorov': self.kolmogorov_test,
            'kuiper': self.kuiper_test,
            'anderson': self.anderson_darling_test,
            'cramer': self.cramer_mise_test,
            'watson': self.watson,
            'neyman': self.neyman_smooth_test,
            'chi_squared': self.chi_squared_test,
            'rao': self.rao_spacing_test,
            'greenwood': self.greenwood_test,
            'sum_based': self.h_opt_gum,
            'sum_based_its': self.h_opt_dif,
            'log': self.log_score,
            'ars': self.ars_score,
            'phi_divergence': self.phi_divergence_test,
            'head_runs': self.head_runs_test,
            'z_score': self.z_score_test,
            'its_neg': self.its_neg,
            'simple_sum': self.simple_sum,
        }

    def perform_statistical_test(self, test_name):
        config = self.config_manager.get_config(test_name)
        test_method = self.test_methods.get(test_name)
        if test_method:
            return test_method(**config)  # Call the method with the unpacked config
        else:
            raise ValueError(f"Unsupported test name: {test_name}")

    def its_neg(self, alpha=0.05, print_out=True):
        """Performs the neg sum test. This is used for inverse transform watermark."""
        assert self.alg == 'ITSEdit', f"Expected self.alg to be 'ITSEdit', but got {self.alg}."
        if print_out:
            print("Working on its_neg sum test")
        generate_samples = lambda size: 1 - np.sqrt(1 - np.random.uniform(0, 1, size))
        simulation_data = generate_samples((self.simulation_size, self.data.shape[1]))
        simulation_outcome = -np.cumsum(simulation_data, axis=1)
        quantile = np.quantile(simulation_outcome, 1 - alpha, axis=0)
        stats_from_data = -np.cumsum(self.data, axis=1)
        results = stats_from_data >= quantile

        return np.mean(results, axis=0), stats_from_data
    
    def simple_sum(self, alpha=0.05, print_out=True):
        """Performs the simple sum test. This is used for SynthID watermark.
            Note that the inpute data should be already converted to uniform distribution.
        """
        assert self.alg == 'EXP', f"Expected self.alg to be 'EXP', but got {self.alg}."
        if print_out:
            print("Working on simple sum test")
        simulation_data = np.random.uniform(0, 1, (self.simulation_size, self.data.shape[1]))
        simulation_outcome = np.cumsum(simulation_data, axis=1)
        quantile = np.quantile(simulation_outcome, 1 - alpha, axis=0)
        stats_from_data = np.cumsum(self.data, axis=1)
        results = stats_from_data >= quantile

        return np.mean(results, axis=0), stats_from_data

    def z_score_test(self, gamma=0.5, alpha=0.05, print_out=True):
        """Performs the Z score test. This is used for 0,1 test."""
        if print_out:
            print("Working on Z score test")
        def compute_z_score(Ys, config_gamma):
            """Compute z-score for the given observed count and total tokens."""
            observed_count = np.sum(Ys, axis=1)
            T = Ys.shape[1]
            expected_count = config_gamma
            numer = observed_count - expected_count * T
            denom = np.sqrt(T * expected_count * (1 - expected_count))
            z = numer / denom
            return z

        z_scores = []
        for length in range(1, self.data.shape[1] + 1):
            z_score = compute_z_score(self.data[:, :length], gamma)
            z_scores.append(z_score)
        stats_from_data = np.array(z_scores).T
        simu_data = np.random.randint(0, 2, size=(self.simulation_size, self.data.shape[1]))
        simu_z_scores = compute_z_score(simu_data, gamma)
        quantile = np.quantile(simu_z_scores, 1 - alpha)
        result = np.mean(stats_from_data >= quantile, axis=0)

        return result, stats_from_data

    def head_runs_test(self, skip=49, alpha=0.05, simulation_size=1000, print_out=True):
        """Performs the Head runs test. This is used for 0,1 test."""
        if print_out:
            print("Working on head runs test")
        def compute_head_runs(sequence):
            head_runs = []
            current_run_length = 0
            # Iterate over the sequence
            for value in sequence:
                if value == 1:
                    current_run_length += 1
                elif current_run_length > 0:  # End of a run of 1s
                    head_runs.append(current_run_length)
                    current_run_length = 0  # Reset for the next possible run
            # Check if the last run of 1s extends to the end of the sequence
            if current_run_length > 0:
                head_runs.append(current_run_length)
            return head_runs

        def get_head_runs_mean(Ys):
            head_runs_means = []
            for sample in Ys:
                head_runs = compute_head_runs(sample)
                head_runs_means.append(np.mean(head_runs))
            return np.array(head_runs_means)

        results = []
        stats_from_data = []
        for length in range(skip + 1, self.data.shape[1] + 1):
            head_runs_means = get_head_runs_mean(self.data[:, :length])
            simu_data = np.random.randint(0, 2, size=(simulation_size, length))
            simu_head_runs_means = get_head_runs_mean(simu_data)
            head_runs_quantile = np.quantile(simu_head_runs_means, 1 - alpha)
            result = np.mean(head_runs_means >= head_runs_quantile)
            results.append(result)
            stats_from_data.append(head_runs_means)
        stats_from_data = np.array(stats_from_data).T
        return np.array(results), stats_from_data

    def phi_divergence_test(self, **kwargs):
        """Performs the Phi divergence based test."""
        if kwargs.get('print_out'):
            print("Working on Phi divergence based test")
        self.phi_divergence.alpha = kwargs.get('alpha', 0.05)
        self.phi_divergence.s = kwargs.get('s', 2)
        self.phi_divergence.mask = kwargs.get('mask', True)
        if self.alg == 'EXP':
            return self.phi_divergence.analyze_y(self.data)
        elif self.alg == 'ITSEdit':
            data_trans = (1 - self.data)**2     # this is equivalent to exam 1-Y follow r^2 or not, phi-divergence detect direction
            return self.phi_divergence.analyze_y(data_trans)
        else:
            raise ValueError(f"Unsupported algorithm: {self.alg} for Phi divergence based test.")

    def ars_score(self, ratio=1, alpha=0.05, print_out=True):
        """Performs the Scott Aaronson's score."""
        assert self.alg == 'EXP', f"Expected self.alg to be 'EXP', but got {self.alg}."

        if print_out:
            print("Working on Scott Aaronson's score")

        def compute_gamma(q, check_point):
            qs = []
            for t in check_point:
                qs.append(gamma.ppf(q=q, a=t))
            return np.array(qs)

        m = self.data.shape[-1]
        given_m = int(ratio * m)
        truncated_Ys = self.data[..., :given_m]
        h_ars_Ys = -np.log(1 - truncated_Ys)
        stats_from_data = np.cumsum(h_ars_Ys, axis=1)
        x = np.arange(1, 1 + m)
        h_ars_qs = compute_gamma(1 - alpha, x)
        results = stats_from_data >= h_ars_qs
        return np.mean(results, axis=0), stats_from_data

    def log_score(self, ratio=1, alpha=0.05, print_out=True):
        """Performs the Log score."""
        assert self.alg == 'EXP', f"Expected self.alg to be 'EXP', but got {self.alg}."

        if print_out:
            print('Working on Log score')

        def compute_gamma(q, check_point):
            qs = []
            for t in check_point:
                qs.append(gamma.ppf(q=q, a=t))
            return np.array(qs)

        m = self.data.shape[-1]
        given_m = int(ratio * m)
        truncated_Ys = self.data[..., :given_m]
        h_ars_Ys = np.log(truncated_Ys)
        stats_from_data = np.cumsum(h_ars_Ys, axis=1)
        x = np.arange(1, 1 + m)
        h_log_qs = compute_gamma(alpha, x)
        results = stats_from_data >= -h_log_qs
        return np.mean(results, axis=0), stats_from_data

    def h_opt_gum(self, delta0=0.2, alpha=0.05, print_out=True):
        """Performs the Sum-based test."""
        assert self.alg == 'EXP', f"Expected self.alg to be 'EXP', but got {self.alg}."
        if print_out:
            print('Working on Sum-based Test')
        def f_opt(r, delta):
            inte_here = np.floor(1 / (1 - delta))
            rest = 1 - (1 - delta) * inte_here
            return np.log(inte_here * r ** (delta / (1 - delta)) + r ** (1 / rest - 1))

        # Compute critical values
        h_ars_Ys = f_opt(self.data, delta0)

        def find_q(N=2500):
            Null_Ys = np.random.uniform(size=(N, self.data.shape[1]))
            Simu_Y = f_opt(Null_Ys, delta0)
            Simu_Y = np.cumsum(Simu_Y, axis=1)
            h_help_qs = np.quantile(Simu_Y, 1 - alpha, axis=0)
            return h_help_qs

        q_lst = []
        for N in [2500] * 10:
            q_lst.append(find_q(N))
        h_help_qs = np.mean(np.array(q_lst), axis=0)

        stats_from_data = np.cumsum(h_ars_Ys, axis=1)
        results = stats_from_data >= h_help_qs
        return np.mean(results, axis=0), stats_from_data

    def h_opt_dif(self, delta=0.1, weight=1e-6, vocab_size=50265, alpha=0.05, model_name="1p3B", print_out=True):
        """Performs the Sum-based test."""
        assert self.alg == 'ITSEdit', f"Expected self.alg to be 'ITSEdit', but got {self.alg}."
        if print_out:
            print('Working on Sum-based Test')
        dif = -np.array(self.data)
        final_Y = dif

        def transform(Y):
            ## weight=1e-6 is used to avoid numerical blow-up
            return np.log(np.maximum(1 + Y / (1 - delta), weight) / np.maximum(1 + Y, 0) / (1 - delta))

        # final_Y = np.log(np.maximum(1+final_Y/(1-delta),0)/np.maximum(1+final_Y,0))
        final_Y = transform(final_Y)
        stats_from_data = np.cumsum(final_Y, axis=1)

        # Compute critical values
        ## We use simulation to compute the critical values
        def find_q(N=1000):
            Null_Ys_U = np.random.uniform(size=(N, dif.shape[1]))
            Null_Ys_pi_s = np.random.randint(low=0, high=vocab_size, size=(N, dif.shape[1]))
            Null_etas = np.array(Null_Ys_pi_s) / (vocab_size - 1)
            null_final_Y = -np.abs(Null_Ys_U - Null_etas)
            null_final_Y = transform(null_final_Y)
            null_cumsum_Ys = np.cumsum(null_final_Y, axis=1)
            h_opt_qs = np.quantile(null_cumsum_Ys, 1 - alpha, axis=0)
            return h_opt_qs

        q_lst = []
        if model_name == "2p7B":
            ## If we use the 2.7B model, we should pay more efforts to control the Type I error
            for N in [500] * 10 + [200, 1000, 2000]:
                q_lst.append(find_q(N))
            h_opt_qs = np.min(np.array(q_lst), axis=0)
        else:
            for N in [500] * 10:
                q_lst.append(find_q(N))
            h_opt_qs = np.mean(np.array(q_lst), axis=0)

        results = (stats_from_data >= h_opt_qs)
        return np.mean(results, axis=0), stats_from_data

    def kolmogorov_test(self, alpha, print_out):
        """Performs the Kolmogorov-Smirnov test."""
        if print_out:
            print('Working on Kolmogorov-Smirnov Test')

        def compute_D(Ys, cdf_function):
            _, n_length = Ys.shape
            Max_Ds = np.zeros_like(Ys)
            for l in range(1, n_length + 1):
                Ys_cut = Ys[:, :l]
                Ys_cut = np.sort(Ys_cut, axis=1)
                Dplus = -cdf_function(Ys_cut) + np.arange(1, 1 + l) / l
                Dminus = cdf_function(Ys_cut) - np.arange(l) / l
                max_Dplus = np.max(Dplus, axis=1)
                max_Dminus = np.max(Dminus, axis=1)
                Max_Ds[:, l - 1] = np.maximum(max_Dplus, max_Dminus)
            return Max_Ds

        # Define the CDF function based on self.alg
        if self.alg == 'EXP':
            cdf_function = lambda x: x  # Uniform distribution CDF
            generate_samples = lambda size: np.random.uniform(0, 1, size)
        elif self.alg == 'ITSEdit':
            cdf_function = lambda x: 1 - (1 - x) ** 2  # CDF for ITSEdit
            generate_samples = lambda size: 1 - np.sqrt(1 - np.random.uniform(0, 1, size))
        else:
            raise ValueError("Unsupported algorithm. Use 'EXP' for uniform or 'ITSEdit' for the custom CDF.")

        stats_from_data = compute_D(self.data, cdf_function)
        simulation_data = generate_samples((self.simulation_size, self.data.shape[1]))
        simulation_outcome = compute_D(simulation_data, cdf_function)
        quantile = np.quantile(simulation_outcome, 1 - alpha, axis=0)
        results = stats_from_data >= quantile
        return np.mean(results, axis=0), stats_from_data

    def kuiper_test(self, alpha=0.05, skip=1, print_out=True):
        """Performs the Kuiper's test."""
        if print_out:
            print('Working on Kuiper’s Test')

        def compute_D(Ys, cdf_function):
            """Computes the Kuiper statistic."""
            _, n_length = Ys.shape
            Max_Ds = np.zeros_like(Ys)
            for l in range(1, n_length + 1):
                Ys_cut = Ys[:, :l]
                Ys_cut = np.sort(Ys_cut, axis=1)
                # Compute D+ and D- using the given CDF function
                Dplus = (np.arange(1, 1 + l) / l) - cdf_function(Ys_cut)
                Dminus = cdf_function(Ys_cut) - (np.arange(0, l) / l)
                max_Dplus = np.max(Dplus, axis=1)
                max_Dminus = np.max(Dminus, axis=1)
                Max_Ds[:, l - 1] = max_Dplus + max_Dminus  # Combine for Kuiper's statistic
            return Max_Ds

        # Define the CDF function and sampling based on self.alg
        if self.alg == 'EXP':
            cdf_function = lambda x: x  # Uniform distribution CDF
            generate_samples = lambda size: np.random.uniform(0, 1, size)  # Uniform samples
        elif self.alg == 'ITSEdit':
            cdf_function = lambda x: 1 - (1 - x) ** 2  # CDF for ITSEdit
            generate_samples = lambda size: 1 - np.sqrt(1 - np.random.uniform(0, 1, size))
        else:
            raise ValueError("Unsupported algorithm. Use 'EXP' for uniform or 'ITSEdit' for the custom CDF.")

        stats_from_data = compute_D(self.data, cdf_function)
        simulation_data = generate_samples((self.simulation_size, self.data.shape[1]))
        simulation_outcome = compute_D(simulation_data, cdf_function)
        quantile = np.quantile(simulation_outcome, 1 - alpha, axis=0)
        results = stats_from_data > quantile
        return np.mean(results, axis=0)[skip:], stats_from_data

    def anderson_darling_test(self, alpha=0.05, print_out=True):
        """Performs the Anderson-Darling test."""
        if print_out:
            print('Working on Anderson-Darling Test')

        def compute_AD(Ys, cdf_function):
            """Computes the Anderson-Darling statistic."""
            n_sample, n_length = Ys.shape
            A = np.zeros_like(Ys)
            for l in range(1, n_length + 1):
                Ys_cut = Ys[:, :l]
                Ys_cut = np.sort(Ys_cut, axis=1)
                # Compute CDF values
                cdf_values = cdf_function(Ys_cut)
                log_cdf_values = np.log(cdf_values)
                log_1_minus_cdf_values = np.log(1 - cdf_values)
                current_S = np.zeros(n_sample)
                for i in range(l):
                    current_S += (log_cdf_values[:, i] + log_1_minus_cdf_values[:, l - 1 - i]) * (2 * i + 1) / l
                A[:, l - 1] = -l - current_S
            return A

        # Define the CDF function and sample generation based on self.alg
        if self.alg == 'EXP':
            cdf_function = lambda x: x  # Uniform distribution CDF
            generate_samples = lambda size: np.random.uniform(0, 1, size)  # Uniform samples
        elif self.alg == 'ITSEdit':
            cdf_function = lambda x: 1 - (1 - x) ** 2  # CDF for ITSEdit
            generate_samples = lambda size: 1 - np.sqrt(1 - np.random.uniform(0, 1, size))
        else:
            raise ValueError("Unsupported algorithm. Use 'EXP' for uniform or 'ITSEdit' for the custom CDF.")

        stats_from_data = compute_AD(self.data, cdf_function)
        simulation_data = generate_samples((self.simulation_size, self.data.shape[1]))
        simulation_outcome = compute_AD(simulation_data, cdf_function)
        quantile = np.quantile(simulation_outcome, 1 - alpha, axis=0)
        results = stats_from_data >= quantile
        return np.mean(results, axis=0), stats_from_data

    def cramer_mise_test(self, alpha=0.05, print_out=True):
        """Performs the Cramér–von Mises test."""
        if print_out:
            print('Working on Cramér–von Mises Test')

        def compute_CM(Ys, cdf_function):
            M, L = Ys.shape
            W2 = np.zeros_like(Ys, dtype=float)
            for k in range(1, L + 1):
                sorted_Ys = np.sort(Ys[:, :k], axis=1)
                cdf_values = cdf_function(sorted_Ys)
                Ysquare_cumsum = np.cumsum(cdf_values ** 2, axis=1)
                x = 2 * np.arange(1, k + 1) - 1
                inner_prod = np.cumsum(cdf_values * x, axis=1) / np.arange(1, k + 1)
                x2 = np.cumsum(x ** 2) / np.arange(1, k + 1) ** 2 / 4
                W2[:, k - 1] = Ysquare_cumsum[:, -1] - inner_prod[:, -1] + x2[-1] + 1 / 12 / np.arange(1, k + 1)[-1]
            return np.sqrt(W2)

        if self.alg == 'EXP':
            cdf_function = lambda x: x
            generate_samples = lambda size: np.random.uniform(0, 1, size)
        elif self.alg == 'ITSEdit':
            cdf_function = lambda x: 1 - (1 - x) ** 2
            generate_samples = lambda size: 1 - np.sqrt(1 - np.random.uniform(0, 1, size))
        else:
            raise ValueError("Unsupported algorithm. Use 'EXP' for uniform or 'ITSEdit' for the custom CDF.")

        stats_from_data = compute_CM(self.data, cdf_function)
        simulation_data = generate_samples((self.simulation_size, self.data.shape[1]))
        simulation_outcome = compute_CM(simulation_data, cdf_function)
        quantile = np.quantile(simulation_outcome, 1 - alpha, axis=0)
        results = stats_from_data >= quantile
        return np.mean(results, axis=0), stats_from_data

    def watson(self, alpha=0.05, skip=1, print_out=True):
        """Performs Watson's test."""
        if print_out:
            print("Working on Watson's Test")

        def compute_WU(Ys, cdf_function):
            M, L = Ys.shape
            U2 = np.zeros_like(Ys, dtype=float)
            for k in range(1, L + 1):
                sorted_Ys = np.sort(Ys[:, :k], axis=1)
                cdf_values = cdf_function(sorted_Ys)
                cdf_square_cumsum = np.cumsum(cdf_values ** 2, axis=1)
                x = 2 * np.arange(1, k + 1) - 1
                inner_prod = np.cumsum(cdf_values * x, axis=1) / np.arange(1, k + 1)
                x2 = np.cumsum(x ** 2) / np.arange(1, k + 1) ** 2 / 4
                W2 = cdf_square_cumsum[:, -1] - inner_prod[:, -1] + x2[-1] + 1 / 12 / np.arange(1, k + 1)[-1]
                U2[:, k - 1] = W2 - k * (np.mean(cdf_values, axis=1) - 0.5) ** 2

            return np.sqrt(U2)

        if self.alg == 'EXP':
            cdf_function = lambda x: x
            generate_samples = lambda size: np.random.uniform(0, 1, size)
        elif self.alg == 'ITSEdit':
            cdf_function = lambda x: 1 - (1 - x) ** 2
            generate_samples = lambda size: 1 - np.sqrt(1 - np.random.uniform(0, 1, size))
        else:
            raise ValueError("Unsupported algorithm. Use 'EXP' for uniform or 'ITSEdit' for the custom CDF.")

        stats_from_data = compute_WU(self.data, cdf_function)
        simulation_data = generate_samples((self.simulation_size, self.data.shape[1]))
        simulation_outcome = compute_WU(simulation_data, cdf_function)
        quantile = np.quantile(simulation_outcome, 1 - alpha, axis=0)
        results = stats_from_data >= quantile
        return np.mean(results, axis=0)[skip:], stats_from_data

    def neyman_smooth_test(self, alpha=0.05, k=3, print_out=True):
        """Performs the Neyman's Smooth test."""
        if print_out:
            print('Working on Neyman Smooth Test')
        M, L = self.data.shape
        critical_value = chi2.ppf(1 - alpha, df=k)

        def orthonormal_h(y, j):
            """Computes the orthonormal polynomial basis for Neyman's Smooth test."""
            if j == 1:
                return np.sqrt(3) * (2 * y - 1)
            elif j == 2:
                return np.sqrt(5) * (6 * y ** 2 - 6 * y + 1)
            elif j == 3:
                return np.sqrt(7) * (20 * y ** 3 - 30 * y ** 2 + 12 * y - 1)
            else:
                raise ValueError("Only supports up to 3 orthonormal polynomials.")

        def compute_statistic(data, cdf_function, length, k):
            """Computes the Neyman test statistic \( T_k \)."""
            a_squares_sum = 0
            cdf_values = cdf_function(data[:length])    # convert data to be uniformly distributed under null.
            for j in range(1, k + 1):
                hj_sum = np.sum(orthonormal_h(cdf_values, j))
                a_squares_sum += hj_sum ** 2
            return a_squares_sum / length

        # Define the CDF function and sampling based on self.alg
        if self.alg == 'EXP':
            cdf_function = lambda x: x
        elif self.alg == 'ITSEdit':
            cdf_function = lambda x: 1 - (1 - x) ** 2
        else:
            raise ValueError("Unsupported algorithm. Use 'EXP' for uniform or 'ITSEdit' for the custom CDF.")

        N = np.zeros(L)
        stats_from_data = []
        for l in range(1, L + 1):
            Y_statistics = np.array([
                compute_statistic(self.data[i, :l], cdf_function, l, k)
                for i in range(M)
            ])
            N[l - 1] = np.sum(Y_statistics >= critical_value) / M
            stats_from_data.append(Y_statistics)
        stats_from_data = np.array(stats_from_data)
        return N, stats_from_data

    def chi_squared_test(self, alpha=0.05, c=10, skip=50, print_out=True):
        """Performs the Chi-Squared test."""
        if print_out:
            print('Working on Chi-Squared Test')

        M, L = self.data.shape
        A = np.zeros(L - skip)
        stats_from_data = []
        critical_value = chi2.ppf(1 - alpha, c - 1)

        # Define the CDF function and sampling based on self.alg
        if self.alg == 'EXP':
            cdf_function = lambda x: x
        elif self.alg == 'ITSEdit':
            cdf_function = lambda x: 1 - (1 - x) ** 2
        else:
            raise ValueError("Unsupported algorithm. Use 'EXP' for uniform or 'ITSEdit' for the custom CDF.")

        # Iterate over each text length starting from skip+1
        for l in range(skip + 1, L + 1):
            observed_frequencies = np.zeros((M, c))
            for i in range(M):
                transformed_data = cdf_function(self.data[i, :l])
                observed_frequencies[i], _ = np.histogram(transformed_data, bins=np.linspace(0, 1, c + 1))
            expected_frequencies = l / c
            chi_squared_statistics = np.sum(
                ((observed_frequencies - expected_frequencies) ** 2) / expected_frequencies,
                axis=1
            )
            stats_from_data.append(chi_squared_statistics)
            A[l - skip - 1] = np.mean(chi_squared_statistics >= critical_value)
        stats_from_data = np.array(stats_from_data)
        return A, stats_from_data

    def rao_spacing_test(self, alpha=0.05, print_out=True):
        """Performs the Rao's Spacing test."""
        if print_out:
            print('Working on Rao Spacing Test')
        def compute_Rao(Ys):
            n_sample, n_length = Ys.shape
            U = np.zeros_like(Ys)
            for l in range(1, n_length + 1):
                Ys_cut = Ys[:, :l]
                Ys_sorted = np.sort(Ys_cut, axis=1)
                spacings = np.diff(Ys_sorted, axis=1)
                S_n = 1 - Ys_sorted[:, -1]
                spacings = np.hstack((spacings, S_n[:, np.newaxis]))
                U[:, l - 1] = 0.5 * l * np.sum(np.abs(spacings - 1 / l), axis=1)
            return U

        if self.alg == 'ITSEdit':
            cdf_function = lambda x: 1 - (1 - x) ** 2
            data = cdf_function(self.data)
        elif self.alg == 'EXP':
            data = self.data
        else:
            raise ValueError("Unsupported algorithm. Use 'EXP' for uniform or 'ITSEdit' for the custom CDF.")
        stats_from_data = compute_Rao(data)
        simulation_data = np.random.uniform(0, 1, (self.simulation_size, self.data.shape[1]))
        simulation_outcome = compute_Rao(simulation_data)
        quantile = np.quantile(simulation_outcome, 1 - alpha, axis=0)
        results = stats_from_data >= quantile

        return np.mean(results, axis=0), stats_from_data

    def greenwood_test(self, alpha=0.05, print_out=True):
        """Performs the Greenwood's test."""
        if print_out:
            print("Working on Greenwood's Test")
        def compute_green(Ys):
            n_sample, n_length = Ys.shape
            G = np.zeros_like(Ys)
            for l in range(1, n_length + 1):
                Ys_cut = Ys[:, :l]
                Ys_sorted = np.sort(Ys_cut, axis=1)
                spacings = np.diff(Ys_sorted, axis=1, prepend=0)
                G[:, l - 1] = l * np.sum(spacings ** 2, axis=1)
            return G

        # Define the sampling function based on self.alg
        # if self.alg == 'EXP':
        #     generate_samples = lambda size: np.random.uniform(0, 1, size)
        # elif self.alg == 'ITSEdit':
        #     generate_samples = lambda size: 1 - np.sqrt(1 - np.random.uniform(0, 1, size))
        # else:
        #     raise ValueError("Unsupported algorithm. Use 'EXP' for uniform or 'ITSEdit' for the custom CDF.")
        if self.alg == 'ITSEdit':
            cdf_function = lambda x: 1 - (1 - x) ** 2
            data = cdf_function(self.data)
        elif self.alg == 'EXP':
            data = self.data
        else:
            raise ValueError("Unsupported algorithm. Use 'EXP' for uniform or 'ITSEdit' for the custom CDF.")
        stats_from_data = compute_green(data)
        # simulation_data = generate_samples((self.simulation_size, self.data.shape[1]))
        simulation_data = np.random.uniform(0, 1, (self.simulation_size, self.data.shape[1]))
        simulation_outcome = compute_green(simulation_data)
        quantile = np.quantile(simulation_outcome, 1 - alpha, axis=0)
        results = stats_from_data >= quantile

        return np.mean(results, axis=0), stats_from_data

    def plot_results(self, test_name_list, type_2=True, grid=False, y_log=False, custom_line_styles=None, custom_colors=None,
                     custom_font_size=12, custom_font_family="Times New Roman", font_path=None, figsize=(10, 6),
                     show_legend=True, save_fig=False, save_path='output_figure.pdf'):
        if save_fig:
            # If saving the figure, use 'Agg' backend to allow file creation without display
            plt.switch_backend('Agg')

        # Load the font if font_path is provided
        if font_path is not None:
            font_manager.fontManager.addfont(font_path)
            prop = font_manager.FontProperties(fname=font_path)
            font_family = prop.get_name()
            print('font_path provided, use the custom font.')   # Use the custom font if font_path is provided
        else:
            font_family = custom_font_family  # Use the default custom font family
        # Set plot styling within a context manager
        with plt.rc_context({
            "font.family": font_family,
            'font.size': custom_font_size,
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsfonts}'
        }):
            plt.figure(figsize=figsize)
            # Use custom styles or default styles
            colors = custom_colors if custom_colors is not None else self.default_colors
            if len(test_name_list) <= len(colors):
                line_styles = ['-']
            else:
                line_styles = custom_line_styles if custom_line_styles is not None else self.default_line_styles

            for index, test_name in enumerate(test_name_list):
                results, _ = self.perform_statistical_test(test_name)
                if type_2:
                    results = 1 - results

                # Apply styles and colors
                line_style = line_styles[index % len(line_styles)]
                color = colors[index % len(colors)]
                if test_name in ['chi_squared', 'kuiper', 'watson', 'head_runs']:
                    skip_len = self.config_manager.get_config(test_name)['skip']
                    x = np.arange(skip_len + 1, skip_len + 1 + results.shape[0])
                elif test_name == 'phi_divergence':
                    x = np.arange(1, 1 + self.text_len, 10)
                else:
                    x = np.arange(1, results.shape[0] + 1)
                plt.plot(x, results, label=f'{test_name}', linestyle=line_style, color=color)

            plt.xlabel('Text length')
            if type_2:
                plt.ylabel('Type II Error')
            else:
                plt.ylabel('Accuracy')
            # plt.title('Classification Results by Statistics')
            if y_log:
                plt.yscale('log')
            if show_legend:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.grid(grid)
            # Save or show the figure based on user input
            if save_fig:
                plt.savefig(save_path, format='pdf', bbox_inches='tight')  # Saves the figure to the path in PDF format
                plt.close()
                print(f"Figure saved as {save_path}")
            else:
                plt.show()


def plot_multi_results(data_list, test_name_list, type_2=True, grid=False, y_log=False, custom_line_styles=None,
                       custom_colors=None, custom_font_size=12, custom_font_family="Times New Roman", font_path=None,
                       figsize=(12, 12), show_legend=None, save_fig=False, save_path='output_multi_figure.pdf'):
    """
    Plot results for multiple datasets on a single figure with subplots and a shared legend.

    :param data_list: List of datasets. Each dataset is a tuple of (name, StatisticalTests instance).
    :param test_name_list: List of test names to plot.
    :param type_2: Whether to plot Type II error (True) or accuracy (False).
    :param grid: Whether to show gridlines.
    :param y_log: Whether to use a logarithmic scale for the y-axis.
    :param custom_line_styles: Custom line styles for the plots.
    :param custom_colors: Custom colors for the plots.
    :param custom_font_size: Font size for the plot.
    :param custom_font_family: Font family for the plot.
    :param font_path: Path to a custom font file.
    :param figsize: Overall figure size.
    :param show_legend: Whether to show a shared legend.
    :param save_fig: Whether to save the figure to a file.
    :param save_path: File path to save the figure.
    """
    if save_fig:
        # If saving the figure, use 'Agg' backend to allow file creation without display
        plt.switch_backend('Agg')
    # Load the font if font_path is provided
    if font_path is not None:
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        font_family = prop.get_name()
        print('font_path provided, using the custom font.')
    else:
        font_family = custom_font_family  # Use the default custom font family

    # Determine grid size (e.g., 2x2 for 4 datasets)
    num_datasets = len(data_list)
    nrows = int(np.ceil(np.sqrt(num_datasets)))
    ncols = int(np.ceil(num_datasets / nrows))

    # Set plot styling within a context manager
    with plt.rc_context({
        "font.family": font_family,
        'font.size': custom_font_size,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    }):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        axes = axes.flatten()  # Flatten axes array for easier indexing
        all_handles = []  # To collect handles for the legend
        all_labels = []  # To collect labels for the legend

        for idx, (dataset_name, stats_instance) in enumerate(data_list):
            ax = axes[idx]
            colors = custom_colors if custom_colors is not None else stats_instance.default_colors
            if len(test_name_list) <= len(colors):
                line_styles = ['-']
            else:
                line_styles = custom_line_styles if custom_line_styles is not None else stats_instance.default_line_styles

            for test_idx, test_name in enumerate(test_name_list):
                results, _ = stats_instance.perform_statistical_test(test_name)
                if type_2:
                    results = 1 - results

                line_style = line_styles[test_idx % len(line_styles)]
                color = colors[test_idx % len(colors)]
                if test_name in ['chi_squared', 'kuiper', 'watson', 'head_runs']:
                    skip_len = stats_instance.config_manager.get_config(test_name)['skip']
                    x = np.arange(skip_len + 1, skip_len + 1 + results.shape[0])
                elif test_name == 'phi_divergence':
                    x = np.arange(1, 1 + stats_instance.text_len, 10)
                else:
                    x = np.arange(1, results.shape[0] + 1)

                handle, = ax.plot(x, results, label=f'{test_name}', linestyle=line_style, color=color)
                # Collect handles and labels only once
                if idx == 0:
                    all_handles.append(handle)
                    all_labels.append(f'{test_name}')

            ax.set_title(dataset_name)
            # ax.set_xlabel('Text length')
            # if type_2:
            #     ax.set_ylabel('Type II Error')
            # else:
            #     ax.set_ylabel('Accuracy')
            if y_log:
                ax.set_yscale('log')
            ax.grid(grid)
            if show_legend == idx:
                ax.legend()
        # Hide unused subplots if any
        for idx in range(len(data_list), len(axes)):
            fig.delaxes(axes[idx])

        # Adding shared labels
        fig.text(0.5, -0.03, 'Watermarked text length', ha='center')
        fig.text(-0.03, 0.5, 'Type II Error', va='center', rotation='vertical')
        # Add a shared legend outside the subplots
        if show_legend == 'figure':
            fig.legend(handles=all_handles, labels=all_labels, loc='center right', bbox_to_anchor=(1.25, 0.5), borderaxespad=0.)
        # Save or show the figure
        if save_fig:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Figure saved as {save_path}")
        else:
            plt.show()




