#coding: utf-8
from typing import Dict,Tuple,Callable

SUPPORTED_COLORINGS:Dict[str,Tuple[str,str]] = {
    "ACCENT"        : ('\x1b[01m', '\x1b[01m'),
    "BLACK"         : ('\x1b[30m', '\x1b[40m'),
    "RED"           : ('\x1b[31m', '\x1b[41m'),
    "GREEN"         : ('\x1b[32m', '\x1b[42m'),
    "YELLOW"        : ('\x1b[33m', '\x1b[43m'),
    "BLUE"          : ('\x1b[34m', '\x1b[44m'),
    "MAGENTA"       : ('\x1b[35m', '\x1b[45m'),
    "CYAN"          : ('\x1b[36m', '\x1b[46m'),
    "WHITE"         : ('\x1b[37m', '\x1b[47m'),
    "DEFAULT"       : ('\x1b[39m', '\x1b[49m'),
    "GRAY"          : ('\x1b[90m', '\x1b[100m'),
    "BRIGHT_RED"    : ('\x1b[91m', '\x1b[101m'),
    "BRIGHT_GREEN"  : ('\x1b[92m', '\x1b[102m'),
    "BRIGHT_YELLOW" : ('\x1b[93m', '\x1b[103m'),
    "BRIGHT_BLUE"   : ('\x1b[94m', '\x1b[104m'),
    "BRIGHT_MAGENTA": ('\x1b[95m', '\x1b[105m'),
    "BRIGHT_CYAN"   : ('\x1b[96m', '\x1b[106m'),
    "BRIGHT_WHITE"  : ('\x1b[97m', '\x1b[107m'),
    # "END"           : ('\x1b[0m',  '\x1b[0m'),
}

def _toCOLOR_create(color:str="") -> Callable[[str,bool],str]:
    color = color.upper()
    charcode = SUPPORTED_COLORINGS[color]
    func = lambda x,is_bg=False: f"{charcode[is_bg]}{str(x)}\x1b[0m"
    return func

toACCENT         = _toCOLOR_create(color="ACCENT")
toBLACK          = _toCOLOR_create(color="BLACK")
toRED            = _toCOLOR_create(color="RED")
toGREEN          = _toCOLOR_create(color="GREEN")
toYELLOW         = _toCOLOR_create(color="YELLOW")
toBLUE           = _toCOLOR_create(color="BLUE")
toMAGENTA        = _toCOLOR_create(color="MAGENTA")
toCYAN           = _toCOLOR_create(color="CYAN")
toWHITE          = _toCOLOR_create(color="WHITE")
toDEFAULT        = _toCOLOR_create(color="DEFAULT")
toGRAY           = _toCOLOR_create(color="GRAY")
toBRIGHT_RED     = _toCOLOR_create(color="BRIGHT_RED")
toBRIGHT_GREEN   = _toCOLOR_create(color="BRIGHT_GREEN")
toBRIGHT_YELLOW  = _toCOLOR_create(color="BRIGHT_YELLOW")
toBRIGHT_BLUE    = _toCOLOR_create(color="BRIGHT_BLUE")
toBRIGHT_MAGENTA = _toCOLOR_create(color="BRIGHT_MAGENTA")
toBRIGHT_CYAN    = _toCOLOR_create(color="BRIGHT_CYAN")
toBRIGHT_WHITE   = _toCOLOR_create(color="BRIGHT_WHITE")