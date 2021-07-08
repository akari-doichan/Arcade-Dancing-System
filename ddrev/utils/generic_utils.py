# coding: utf-8
import argparse
import datetime
import re
from typing import Optional


def str_strip(string):
    """Convert all consecutive whitespace  characters to `' '` (half-width whitespace), then return a copy of the string with leading and trailing whitespace removed.

    Args:
        string (str) : string

    Example:
        >>> from ddrev.utils import str_strip
        >>> str_strip(" hoge   ")
        'hoge'
        >>> str_strip(" ho    ge   ")
        'ho ge'
        >>> str_strip("  ho    g　e")
        'ho g e'
    """
    return re.sub(pattern=r"[\s 　]+", repl=" ", string=str(string)).strip()


class ListParamProcessor(argparse.Action):
    """Receive List arguments.

    Examples:
        >>> import argparse
        >>> from ddrev.utils import ListParamProcessor
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--list_params", action=ListParamProcessor)
        >>> args = parser.parse_args(args=["--list_params", "[あ, い, う]"])
        >>> args.list_params
        ['あ', 'い', 'う']

    Note:
        If you run from the command line, execute as follows::

        $ python app.py --list_params "[あ, い, う]"

    """

    def __call__(self, parser, namespace, values, option_strings=None, **kwargs):
        match = re.match(pattern=r"(?:\[|\()(.+)(?:\]|\))", string=values)
        if match:
            values = [int(str_strip(e)) for e in match.group(1).split(",")]
        else:
            values = [int(values)]
        setattr(namespace, self.dest, values)


def now_str(
    tz: Optional[datetime.timezone] = None, fmt: str = "%Y-%m-%d@%H.%M.%S"
) -> str:
    """Returns new datetime string representing current time local to ``tz`` under the control of an explicit format string.

    Args:
        tz (Optional[datetime.timezone], optional): Timezone object. If no ``tz`` is specified, uses local timezone. Defaults to ``None``.
        fmt (str, optional)                       : A format string. See `Python Documentation <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_. Defaults to ``"%Y-%m-%d@%H.%M.%S"``.

    Returns:
        str: A datetime string representing current time local to ``tz``.

    Example:
        >>> from ddrev.utils import now_str
        >>> now_str()
        '2020-09-14@22.31.17'
        >>> now_str(fmt="%A, %d. %B %Y %I:%M%p")
        Monday, 14. September 2020 10:31PM'
        >>> now_str(tz=datetime.timezone.utc)
        '2020-09-14@13.31.17'
    """
    return datetime.datetime.now(tz=tz).strftime(fmt)
