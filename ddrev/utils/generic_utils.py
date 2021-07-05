# coding: utf-8

import datetime
from typing import Optional


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
