# Estimators

Each horizon starts from the estimator's state estimate. Every estimator implements one hook, `estimate(state, u_prev, measurement, args)`, over a state of its own, whose `plant` field is the estimate.

::: diff_soup.optimization.control.Estimator
    options:
      members:
        - init
        - estimate

---

::: diff_soup.optimization.control.FullState
    options:
      members: false

::: diff_soup.optimization.control.PlantEstimate
    options:
      members: false

---

::: diff_soup.optimization.control.ExtendedKalmanFilter
    options:
      members: false

::: diff_soup.optimization.control.KalmanEstimate
    options:
      members: false

---

::: diff_soup.optimization.control.MovingHorizonEstimator
    options:
      members: false

::: diff_soup.optimization.control.MovingHorizonEstimate
    options:
      members: false
