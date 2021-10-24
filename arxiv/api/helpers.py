from aenum import Enum


class Status(Enum):
    ok = 0
    error = 1
    calculating = 2


def get_return_message(status: Status, msg: str, data=None):
    if data is None:
        data = {}
    return {"status": str(status), "msg": msg, "data": data}