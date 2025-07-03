from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

def create_custom_progress_bar(
    console: Console = None,  # type: ignore
    color: str = "cyan",
    show_time: bool = True,
    show_spinner: bool = True,
    spinner_style: str = "dots",
    disable=False,
) -> Progress:
    """
    Create a custom progress bar using Rich, optionally including loss reporting.

    :param description: Description of the task
    :param total: Total number of steps
    :param console: Rich Console object (if None, a new one will be created)
    :param color: Color of the progress bar
    :param show_time: Whether to show the time remaining
    :param show_spinner: Whether to show a spinner
    :param spinner_style: Style of the spinner (e.g., "dots", "dots12", "line", "arrow")
    :param show_loss: Whether to show loss information
    :return: A Rich Progress object and task ID
    """
    if console is None:
        console = Console()
    columns = []

    if show_spinner:
        columns.append(SpinnerColumn(spinner_name=spinner_style, style=color))

    columns.extend(
        [
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None, style=color, complete_style=f"bold {color}"),
            TaskProgressColumn(),
            TextColumn("[bold yellow]Loss: {task.fields[loss]:.4f}", justify="right"),
        ]
    )

    if show_time:
        columns.append(TimeRemainingColumn())

    progress = Progress(*columns, console=console, expand=True, disable=disable)
    return progress
