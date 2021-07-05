# coding: utf-8
__all__ = ["ArcadeDancingSystemImprementationError", "KeyError"]


class ArcadeDancingSystemImprementationError(Exception):
    """
    Warnings that developers will resolve.
    Developers are now solving in a simple stupid way.
    """


class KeyError(KeyError):
    """Overwrite original KeyError so that coloring can be used when outputting an error message"""

    def __str__(self):
        return ", ".join(self.args)
