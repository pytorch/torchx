---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Airflow

For pipelines that support Python based execution you can directly use the
TorchX API. TorchX is designed to be easily integrated in to other applications
via the programmatic API. No special Airflow integrations are needed.

With TorchX, you can use Airflow for the pipeline orchestration and run your
PyTorch application (i.e. distributed training) on a remote GPU cluster.

```python
import datetime
import pendulum

from airflow.utils.state import DagRunState, TaskInstanceState
from airflow.utils.types import DagRunType
from airflow.models.dag import DAG
from airflow.decorators import task


DATA_INTERVAL_START = pendulum.datetime(2021, 9, 13, tz="UTC")
DATA_INTERVAL_END = DATA_INTERVAL_START + datetime.timedelta(days=1)
```

To launch a TorchX job from Airflow you can create a Airflow Python task to
import the runner, launch the job and wait for it to complete. If you're running
on a remote cluster you may need to use the virtualenv task to install the
`torchx` package.

```python
@task(task_id=f'hello_torchx')
def run_torchx(message):
    """This is a function that will run within the DAG execution"""
    from torchx.runner import get_runner
    with get_runner() as runner:
        # Run the utils.sh component on the local_cwd scheduler.
        app_id = runner.run_component(
            "utils.sh",
            ["echo", message],
            scheduler="local_cwd",
        )

        # Wait for the the job to complete
        status = runner.wait(app_id, wait_interval=1)

        # Raise_for_status will raise an exception if the job didn't succeed
        status.raise_for_status()

        # Finally we can print all of the log lines from the TorchX job so it
        # will show up in the workflow logs.
        for line in runner.log_lines(app_id, "sh", k=0):
            print(line, end="")
```

Once we have the task defined we can put it into a Airflow DAG and run it like
normal.

```python
from torchx.schedulers.ids import make_unique

with DAG(
    dag_id=make_unique('example_python_operator'),
    schedule=None,
    start_date=DATA_INTERVAL_START,
    catchup=False,
    tags=['example'],
) as dag:
    run_job = run_torchx("Hello, TorchX!")


dagrun = dag.create_dagrun(
    state=DagRunState.RUNNING,
    execution_date=DATA_INTERVAL_START,
    data_interval=(DATA_INTERVAL_START, DATA_INTERVAL_END),
    start_date=DATA_INTERVAL_END,
    run_type=DagRunType.MANUAL,
)
ti = dagrun.get_task_instance(task_id="hello_torchx")
ti.task = dag.get_task(task_id="hello_torchx")
ti.run(ignore_ti_state=True)
assert ti.state == TaskInstanceState.SUCCESS
```

If all goes well you should see `Hello, TorchX!` printed above.

## Next Steps

* Checkout the [runner API documentation](../runner.rst) to learn more about
  programmatic usage of TorchX
* Browse through the collection of [builtin components](../components/overview.rst)
  which can be used in your Airflow pipeline
