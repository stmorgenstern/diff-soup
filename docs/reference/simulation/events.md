# Events and operating protocols

Events interrupt a solve, either stopping it before the end of the interval
or rewriting its state and continuing. See
[events](../../overview/simulation/events.md).

::: diff_soup.Event
    options:
      members: false

::: diff_soup.timestepping.EventSet
    options:
      members: false

::: diff_soup.timestepping.EventObservable
    options:
      members:
        - crosses
        - below
        - above
        - soc
        - voltage

::: diff_soup.timestepping.terminate_steady_state

::: diff_soup.timestepping.Bisection
    options:
      members: false

---

### Operating protocols

A `Schedule` of `Segment`s runs an experiment through rows of the operating
table in turn, and `ControlState` carries the controls through a solve
([controls](../../overview/modeling/controls.md)).

::: diff_soup.Schedule
    options:
      members: false

::: diff_soup.Segment
    options:
      members: false

::: diff_soup.ControlState
    options:
      members:
        - values
        - epoch
        - mode
        - target
        - is_operating
        - operating
        - target_at
        - bump
        - with_values
        - with_operating
