import numpy as np
from numba import njit
import typing as tp
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score


class RNN:

    def __init__(self, C: np.ndarray, evolution_func: tp.Callable, *args, **kwargs):
        self.C = C
        self.N = C.shape[0]
        self.u = kwargs.pop('u_init', np.zeros((self.N,)))
        self.du = np.zeros_like(self.u)
        self.t = kwargs.pop('t_init', 0.0)
        self.net_update_autonomous = kwargs.pop('evolution_func_autonomous', evolution_func)
        self.net_update = evolution_func
        self.func_kwargs = kwargs
        self.func_args = args
        self.state_records = []
        self.readouts = {}
        self.readout_id = 0

    def ridge_fit(self, X: np.ndarray, y: np.ndarray, k: int = 1, verbose: bool = True, readout_key: tp.Any = None,
                  **kwargs):

        if readout_key is None:
            readout_key = self.readout_id
            self.readout_id += 1

        # perform ridge regression
        if k > 1:
            splitter = StratifiedKFold(n_splits=k)
            scores, coefs = [], []
            for i, (train_idx, test_idx) in enumerate(splitter.split(X=X, y=y)):
                classifier = Ridge(**kwargs)
                classifier.fit(X[train_idx], y[train_idx])
                scores.append(classifier.score(X=X[test_idx], y=y[test_idx]))
                coefs.append(classifier.coef_)
        else:
            classifier = Ridge(**kwargs)
            classifier.fit(X, y)
            scores = [classifier.score(X=X, y=y)]
            coefs = [classifier.coef_]

        # store readout weights
        w_out = np.mean(coefs, axis=0)
        self.readouts[readout_key] = w_out if len(w_out.shape) == 1 else w_out.T

        if verbose:
            print(f'Finished readout training. The readout weights are stored under the key: {readout_key}. '
                  f'Please use that key when calling `RNN.test()` or `RNN.predict()`.')
            avg_score = np.mean(scores)
            if k > 1:
                print(f'Average, cross-validated classification performance across {k} test folds: {avg_score}')
            else:
                print(f'Classification performance on training data: {avg_score}')

        return readout_key, scores, coefs

    def ridge_cv_fit(self, X: np.ndarray, y: np.ndarray, readout_key: tp.Any = None, verbose: bool = True, **kwargs):

        if readout_key is None:
            readout_key = self.readout_id
            self.readout_id += 1

        # perform ridge regression
        classifier = RidgeCV(**kwargs)
        classifier.fit(X, y)

        # store readout weights
        w_out = classifier.coef_
        self.readouts[readout_key] = w_out if len(w_out.shape) == 1 else w_out.T
        if verbose:
            print(f'Finished readout training. The readout weights are stored under the key: {readout_key}. '
                  f'Please use that key when calling `RNN.test()` or `RNN.predict()`.')

        return readout_key, classifier

    def test(self, X: np.ndarray, y: np.ndarray, readout_key: tp.Any = None):

        y_predict = self.predict(X, readout_key)
        return r2_score(y, y_predict), y_predict

    def predict(self, X: np.ndarray, readout_key):

        if readout_key is None:
            readout_key = self.readout_id

        return X @ self.readouts[readout_key]

    def run(self, T: float, dt: float, dts: float, outputs: tuple = None, t_init: float = 0.0,
            inp: tp.Optional[np.ndarray] = None, W_in: tp.Optional[np.ndarray] = None, cutoff: float = 0.0):

        if not outputs:
            outputs = (np.arange(0, len(self.u)),)

        steps = int(np.round(T/dt))
        sampling_steps = int(np.round(dts/dt))
        store_steps = int(np.round((T-cutoff)/dts))
        start_step = steps - store_steps*sampling_steps
        self.t += t_init
        self.func_kwargs['dt'] = dt
        self.state_records = [np.zeros((store_steps, len(out))) for out in outputs]
        state_buffers = [np.zeros((len(out), sampling_steps)) for out in outputs]

        if inp is None:

            sample = 0
            rhs_func, u, args, kwargs, results = self.net_update_autonomous, self.u, self.func_args, self.func_kwargs, \
                                                 self.state_records
            for step in range(steps):
                u = rhs_func(u, *args, **kwargs)
                buffer_step = step % sampling_steps
                store_results = step > start_step and buffer_step == 0
                for i, (out, buffer) in enumerate(zip(outputs, state_buffers)):
                    buffer[:, buffer_step] = u[out]
                    if store_results:
                        results[i][sample, :] = np.mean(buffer, axis=1)
                if store_results:
                    sample += 1

        else:

            sample = 0
            rhs_func, u, args, kwargs, results = self.net_update, self.u, self.func_args, self.func_kwargs, \
                                                 self.state_records
            for step in range(steps):
                u = rhs_func(u, np.asarray(inp[:, step], order='C'), W_in, *args, **kwargs)
                buffer_step = step % sampling_steps
                store_results = step > start_step and buffer_step == 0
                for i, (out, buffer) in enumerate(zip(outputs, state_buffers)):
                    buffer[:, buffer_step] = u[out]
                    if store_results:
                        results[i][sample, :] = np.mean(buffer, axis=1)
                if store_results:
                    sample += 1

        print('Finished simulation. The state recordings are available under `state_records`.')
        return results

    def get_coefs(self, key: tp.Any):
        return self.readouts[key].coef_

    def get_scores(self, key: tp.Any):
        return self.readouts[key].cv_values

    def get_classifier(self, key: tp.Any):
        return self.readouts[key]


# class QIFExpAddRNN(RNN):
#
#     def __init__(self, C: np.ndarray, eta: float, J: float, *args, Delta: float = 2.0, tau: float = 1.0,
#                  alpha: float = 0.05, tau_a: float = 10.0, v_th: float = 1e2, **kwargs):
#
#         @njit
#         def qif_update(u: np.ndarray, inp: np.ndarray, W_in: np.ndarray, N: int, C: np.ndarray,
#                        etas: np.ndarray, J: float, tau: float, alpha: float, tau_a: float, v_th: float,
#                        spike_t: np.ndarray, wait: np.ndarray, dt: float = 1e-4) -> tp.Tuple[np.ndarray, np.ndarray]:
#             """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
#             background excitabilities and mono-exponential synaptic depression."""
#
#             # extract state variables from u
#             v, e = u[:N], u[N:]
#
#             # update spike observations
#             wait[:] = np.maximum(wait - 1, 0)
#             mask = 1.*(np.abs(wait) < 0.4)
#             spike_n = 1.0 * ((spike_t > 0.0) * (spike_t <= 1.0))
#             spike_t[:] = np.maximum(spike_t-1, 0)
#
#             # calculate network input
#             rates = spike_n / dt
#             s = rates @ C
#             net_inp = J*s[0, :]*(1-e)
#             ext_inp = W_in @ inp
#
#             # calculate state vector updates
#             v += dt * mask * ((v**2 + etas + ext_inp) / tau + net_inp)
#             e += dt * mask * (alpha*s[0, :] - e/tau_a)
#
#             # calculate new spikes
#             spikes = v > v_th
#             wait[spikes] = (2 * tau / v[spikes] - 6 * (etas + net_inp)[spikes] / v[spikes]**3) / dt
#             spike_t[:, spikes] = dt*tau/v[spikes]
#             v[spikes] = -v[spikes]
#
#             u[:N] = v
#             u[N:] = e
#             return u, spike_n[0, :]
#
#         if "u_init" not in kwargs:
#             kwargs["u_init"] = np.zeros((2*C.shape[0],))
#         super().__init__(C=C, evolution_func=qif_update, *args, **kwargs)
#
#         etas = eta+Delta*np.tan((np.pi/2)*(2.*np.arange(1, self.N+1)-self.N-1)/(self.N+1))
#         spike_t = np.zeros((1, self.N))
#         wait = np.ones((self.N,))
#         self.func_args = (C.shape[0], C, etas, J, tau, alpha, tau_a, v_th, spike_t, wait)


class QIFExpAddSyns(RNN):

    def __init__(self, C: np.ndarray, eta: float, J: float, *args, Delta: float = 0.5, tau: float = 1.0,
                 alpha: float = 0.05, tau_a: float = 10.0, tau_s: float = 1.0, v_th: float = 1e2, **kwargs):

        @njit
        def qif_update(u: np.ndarray, inp: np.ndarray, W_in: np.ndarray, N: int, C: np.ndarray,
                       etas: np.ndarray, J: float, tau: float, alpha: float, tau_a: float, tau_s: float, v_th: float,
                       spike_t: np.ndarray, wait: np.ndarray, dt: float = 1e-4) -> np.ndarray:
            """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
            background excitabilities and mono-exponential synaptic depression."""

            # extract state variables from u
            v, a, x = u[:N], u[N:2*N], u[2*N:3*N]

            # update spike observations
            wait[:] = np.maximum(wait - 1, 0)
            mask = 1.*(np.abs(wait) < 0.4)
            spike_n = 1.0 * ((spike_t > 0.0) * (spike_t <= 1.0))
            spike_t[:] = np.maximum(spike_t-1, 0)

            # calculate network input
            rates = spike_n / dt
            s = rates @ C
            net_inp = J*s[0, :]*tau
            ext_inp = W_in @ inp

            # calculate state vector updates
            v += dt * mask * (v**2 + etas + ext_inp + x - a)/tau
            a += dt * (alpha*s[0, :] - a/tau_a)
            x += dt * (net_inp - x/tau_s)

            # calculate new spikes
            spikes = v > v_th
            wait[spikes] = (2 * tau / v[spikes] - 6 * (etas + net_inp)[spikes] / v[spikes]**3) / dt
            spike_t[:, spikes] = dt*tau/v[spikes]
            v[spikes] = -v[spikes]

            u[:N] = v
            u[N:2*N] = a
            u[2*N:3*N] = x
            u[3*N:] = rates

            return u

        @njit
        def qif_update2(u: np.ndarray, N: int, C: np.ndarray, etas: np.ndarray, J: float, tau: float, alpha: float,
                        tau_a: float, tau_s: float, v_th: float, spike_t: np.ndarray, wait: np.ndarray, dt: float = 1e-4
                        ) -> np.ndarray:
            """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
            background excitabilities and mono-exponential synaptic depression."""

            # extract state variables from u
            v, a, x = u[:N], u[N:2*N], u[2*N:3*N]

            # update spike observations
            wait[:] = np.maximum(wait - 1, 0)
            mask = 1. * (np.abs(wait) < 0.4)
            spike_n = 1.0 * ((spike_t > 0.0) * (spike_t <= 1.0))
            spike_t[:] = np.maximum(spike_t - 1, 0)

            # calculate network input
            rates = spike_n / dt
            s = rates @ C
            net_inp = J * s[0, :] * tau

            # calculate state vector updates
            v += dt * mask * (v ** 2 + etas + x - a) / tau
            a += dt * (alpha * s[0, :] - a / tau_a)
            x += dt * (net_inp - x / tau_s)

            # calculate new spikes
            spikes = v > v_th
            wait[spikes] = (2 * tau / v[spikes] - 6 * (etas + net_inp)[spikes] / v[spikes] ** 3) / dt
            spike_t[:, spikes] = dt * tau / v[spikes]
            v[spikes] = -v[spikes]

            u[:N] = v
            u[N:2*N] = a
            u[2*N:3*N] = x
            u[3*N:] = rates

            return u

        if "u_init" not in kwargs:
            kwargs["u_init"] = np.zeros((4*C.shape[0],))
        kwargs['evolution_func_autonomous'] = qif_update2
        super().__init__(C=C, evolution_func=qif_update, *args, **kwargs)

        etas = eta+Delta*np.tan((np.pi/2)*(2.*np.arange(1, self.N+1)-self.N-1)/(self.N+1))
        spike_t = np.zeros((1, self.N))
        wait = np.ones((self.N,))
        self.func_args = (C.shape[0], C, etas, J, tau, alpha, tau_a, tau_s, v_th, spike_t, wait)


class QIFExpAddNoiseSyns(RNN):

    def __init__(self, C: np.ndarray, eta: float, J: float, *args, Delta: float = 0.5, tau: float = 1.0,
                 alpha: float = 0.05, tau_a: float = 10.0, tau_s: float = 0.5, v_th: float = 1e2, D: float = 2.0,
                 **kwargs):

        @njit
        def qif_update(u: np.ndarray, inp: np.ndarray, W_in: np.ndarray, N: int, C: np.ndarray,
                       etas: np.ndarray, J: float, tau: float, alpha: float, tau_a: float, tau_s: float, v_th: float,
                       D:float, spike_t: np.ndarray, wait: np.ndarray, dt: float = 1e-4
                       ) -> np.ndarray:
            """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
            background excitabilities and mono-exponential synaptic depression."""

            # extract state variables from u
            v, a, x = u[:N], u[N:2*N], u[2*N:]

            # update spike observations
            wait[:] = np.maximum(wait - 1, 0)
            mask = 1.*(np.abs(wait) < 0.4)
            spike_n = 1.0 * ((spike_t > 0.0) * (spike_t <= 1.0))
            spike_t[:] = np.maximum(spike_t-1, 0)

            # calculate network input
            rates = spike_n / dt
            s = rates @ C
            net_inp = J*s[0, :]
            ext_inp = W_in @ inp

            # calculate state vector updates
            v += dt * mask * ((v**2 + etas - a + ext_inp + np.sqrt(D)*np.random.randn(N)/np.sqrt(dt))/tau + x)
            a += dt * (alpha*s[0, :] - a/tau_a)
            x += dt * (net_inp - x/tau_s)

            # calculate new spikes
            spikes = v > v_th
            wait[spikes] = (2 * tau / v[spikes] - 6 * (etas + net_inp)[spikes] / v[spikes]**3) / dt
            spike_t[:, spikes] = dt*tau/v[spikes]
            v[spikes] = -v[spikes]

            u[:N] = v
            u[N:2*N] = a
            u[2*N:3*N] = x
            u[3*N:] = rates

            return u

        if "u_init" not in kwargs:
            kwargs["u_init"] = np.zeros((3*C.shape[0],))
        super().__init__(C=C, evolution_func=qif_update, *args, **kwargs)

        etas = eta+Delta*np.tan((np.pi/2)*(2.*np.arange(1, self.N+1)-self.N-1)/(self.N+1))
        spike_t = np.zeros((1, self.N))
        wait = np.ones((self.N,))
        self.func_args = (C.shape[0], C, etas, J, tau, alpha, tau_a, tau_s, v_th, D, spike_t, wait)


class QIFExpAddNoiseSTDP(RNN):

    def __init__(self, C: np.ndarray, eta: float, J: float, *args, Delta: float = 0.5, tau: float = 1.0,
                 alpha: float = 0.05, tau_a: float = 10.0, v_th: float = 1e2, D: float = 2.0, beta: float = 0.01,
                 tau_e: float = 1.0, gamma_p: float = 1e-4, gamma_n: float = 1e-3, **kwargs):

        @njit
        def qif_update(u: np.ndarray, inp: np.ndarray, W_in: np.ndarray, N: int, C: np.ndarray,
                       etas: np.ndarray, J: float, tau: float, alpha: float, tau_a: float, v_th: float, D:float,
                       beta: float, tau_e: float, gamma_p: float, gamma_n: float, C_in, C_out, spike_t: np.ndarray,
                       wait: np.ndarray, dt: float = 1e-4) -> np.ndarray:
            """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
            background excitabilities and mono-exponential synaptic depression."""

            # extract state variables from u
            v, a, e = u[:N], u[N:2*N], u[2*N:3*N]

            # update spike observations
            wait[:] = np.maximum(wait - 1, 0)
            mask = 1.*(np.abs(wait) < 0.4)
            spike_n = 1.0 * ((spike_t > 0.0) * (spike_t <= 1.0))
            spike_t[:] = np.maximum(spike_t-1, 0)

            # calculate network input
            rates = spike_n / dt
            s = rates @ C
            net_inp = J*s[0, :]
            ext_inp = W_in @ inp

            # calculate state vector updates
            v += dt * mask * ((v**2 + etas - a + ext_inp + np.sqrt(D)*np.random.randn(N)/np.sqrt(dt))/tau + net_inp)
            a += dt * (alpha*s[0, :] - a/tau_a)

            # update synaptic weights
            for idx in np.argwhere(spike_n > 0):
                idx2 = C[:, idx[1]] > 0
                C[idx2, idx[1]] += gamma_p*e[idx2]
                C[idx[1], :] = np.maximum(0, C[idx[1], :] - gamma_n*e)
            C_in_new = np.sum(C, axis=0).reshape((1, N))
            C *= C_in/C_in_new
            #C_out_new = np.sum(C, axis=1).reshape((N, 1))
            #C *= C_out / C_out_new

            # update spike traces
            e += dt * (beta * rates[0, :] - e / tau_e)

            # calculate new spikes
            spikes = v > v_th
            wait[spikes] = (2 * tau / v[spikes] - 6 * (etas + net_inp)[spikes] / v[spikes]**3) / dt
            spike_t[:, spikes] = dt*tau/v[spikes]
            v[spikes] = -v[spikes]

            u[:N] = v
            u[N:2*N] = a
            u[2*N:3*N] = e
            u[3*N:] = rates
            return u

        if "u_init" not in kwargs:
            kwargs["u_init"] = np.zeros((3*C.shape[0],))
        super().__init__(C=C, evolution_func=qif_update, *args, **kwargs)

        etas = eta+Delta*np.tan((np.pi/2)*(2.*np.arange(1, self.N+1)-self.N-1)/(self.N+1))
        spike_t = np.zeros((1, self.N))
        wait = np.ones((self.N,))
        self.func_args = (self.N, C, etas, J, tau, alpha, tau_a, v_th, D, beta, tau_e, gamma_p, gamma_n,
                          np.sum(C, axis=0).reshape((1, self.N)), np.sum(C, axis=1).reshape((self.N, 1)), spike_t, wait)


class mQIFExpAddSynsRNN(RNN):

    def __init__(self, C: np.ndarray, eta: tp.Union[np.ndarray, float], J: tp.Union[np.ndarray, float],
                 Delta: tp.Union[np.ndarray, float], tau: tp.Union[np.ndarray, float],
                 alpha: tp.Union[np.ndarray, float], tau_a: tp.Union[np.ndarray, float],
                 tau_s: tp.Union[np.ndarray, float], *args,  **kwargs):

        @njit
        def mqif_update(u: np.ndarray, inp: np.ndarray, W_in: np.ndarray, N: int, C: np.ndarray, eta: np.ndarray,
                        Delta: np.ndarray, J: np.ndarray, tau: np.ndarray, alpha: np.ndarray, tau_a: np.ndarray,
                        tau_s: np.ndarray, dt: float = 1e-4) -> np.ndarray:
            """Calculates right-hand side update of a network of coupled QIF populations
            (described by mean-field equations) with heterogeneous background excitabilities and mono-exponential
            synaptic depression."""

            # extract state variables from u
            v, r, a, x = u[:N], u[N:2*N], u[2*N:3*N], u[3*N:]

            # calculate network input
            s = C @ r
            net_inp = J*s[0, :]*tau
            ext_inp = W_in @ inp

            # calculate state vector updates
            r += dt * (Delta/(tau*np.pi) + 2*r*v)
            v += dt * (v**2 + eta + ext_inp + x - a) / tau
            a += dt * (alpha*s[0, :] - a/tau_a)
            x += dt * (net_inp - x/tau_s)

            u[:N] = v
            u[N:2*N] = r
            u[2*N:3*N] = a
            u[3*N:] = x
            return u

        @njit
        def mqif_update2(u: np.ndarray, N: int, C: np.ndarray, eta: np.ndarray, Delta: np.ndarray, J: np.ndarray,
                         tau: np.ndarray, alpha: np.ndarray, tau_a: np.ndarray, tau_s: np.ndarray, dt: float = 1e-4
                         ) -> np.ndarray:
            """Calculates right-hand side update of a network of coupled QIF populations
            (described by mean-field equations) with heterogeneous background excitabilities and mono-exponential
            synaptic depression."""

            # extract state variables from u
            v, r, a, x = u[:N], u[N:2 * N], u[2 * N:3 * N], u[3 * N:]

            # calculate network input
            s = C @ r
            net_inp = J*s*tau

            # calculate state vector updates
            r += dt * (Delta / (tau * np.pi) + 2*r*v) / tau
            v += dt * (v**2 + eta + x - a - (tau*np.pi*r)**2) / tau
            a += dt * (alpha*s - a / tau_a)
            x += dt * (net_inp - x / tau_s)

            u[:N] = v
            u[N:2 * N] = r
            u[2 * N:3 * N] = a
            u[3 * N:] = x
            return u

        if "u_init" not in kwargs:
            kwargs["u_init"] = np.zeros((4*C.shape[0],))
        kwargs['evolution_func_autonomous'] = mqif_update2
        super().__init__(C=C, evolution_func=mqif_update, *args, **kwargs)
        self.func_args = (C.shape[0], C, eta, Delta, J, tau, alpha, tau_a, tau_s)
