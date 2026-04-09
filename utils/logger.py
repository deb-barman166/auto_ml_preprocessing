"""
utils/logger.py  –  Structured, colourised logging for the engine.
"""
import logging, sys

_R="\033[0m"; _B="\033[1m"; _G="\033[32m"; _Y="\033[33m"
_RE="\033[31m"; _C="\033[36m"

class _CF(logging.Formatter):
    _LC = {logging.DEBUG:_C, logging.INFO:_G,
           logging.WARNING:_Y, logging.ERROR:_RE+_B}
    _FMT = "%(asctime)s  [{lc}%(levelname)-8s"+_R+"]  ["+_C+"%(name)s"+_R+"]  %(message)s"
    def format(self, rec):
        lc = self._LC.get(rec.levelno, "")
        fmt = logging.Formatter(self._FMT.format(lc=lc), datefmt="%Y-%m-%d %H:%M:%S")
        return fmt.format(rec)

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(level); h.setFormatter(_CF())
    log.addHandler(h); log.propagate = False
    return log

def get_file_logger(name:str, filepath:str="preprocessing.log") -> logging.Logger:
    log = get_logger(name, logging.DEBUG)
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        fh = logging.FileHandler(filepath, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  [%(levelname)-8s]  [%(name)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        log.addHandler(fh)
    return log
