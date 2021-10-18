# Hello World

This is a self contained guide on how to build a simple app and component spec
and launch it via two different schedulers.

See the [Quickstart](../quickstart.rst) for how to install TorchX locally before
following this example.

Lets start off with writing a simple "Hello World" python app. This is just a
normal python program and can contain anything you'd like.

> **NOTE**: This example uses Jupyter Notebook `%%writefile` to create local files for
example purposes. Under normal usage you would have these as standalone files.

```python
%%writefile my_app.py

import sys
import argparse

def main(user: str) -> None:
    print(f"Hello, {user}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hello world app"
    )
    parser.add_argument(
        "--user",
        type=str,
        help="the person to greet",
        required=True,
    )
    args = parser.parse_args(sys.argv[1:])

    main(args.user)
```

Now that we have an app we can write the component file for it. This
function allows us to reuse and share our app in a user friendly way.

We can use this component from the `torchx` cli or programmatically as part of a
pipeline.

```python
%%writefile my_component.py

import torchx.specs as specs

def greet(user: str, image: str = "my_app:latest") -> specs.AppDef:
    """
    Echos a message to stdout (calls /bin/echo)

    Args:
        user: name of the person to greet
        image: image to use
    """
    return specs.AppDef(
        name="hello_world",
        roles=[
            specs.Role(
                name="greeter",
                image=image,
                entrypoint="python",
                args=[
                    "-m", "my_app",
                    "--user", user,
                ],
            )
        ],
    )
```

Once we write our component, we can then call it via `torchx run`. The
`local_cwd` scheduler executes the component relative to the current directory.

```sh
torchx run --scheduler local_cwd my_component.py:greet --user "your name"
```

If we want to run in other environments, we can build a Docker container so we
can run our component in Docker enabled environments such as Kubernetes or via
the local Docker scheduler.

> **NOTE:** this requires Docker installed and won't work in environments such as
Google Colab.

```python
%%writefile Dockerfile

FROM ghcr.io/pytorch/torchx:0.1.0rc1

ADD my_app.py .
```

Once we have the Dockerfile created we can create our docker image.

```sh
docker build -t my_app:latest -f Dockerfile .
```

We can then launch it on the local scheduler.

```sh
torchx run --scheduler local_docker my_component.py:greet --image "my_app:latest" --user "your name"
```

If you have a Kubernetes cluster you can use the Kubernetes scheduler to launch
this on the cluster instead.
