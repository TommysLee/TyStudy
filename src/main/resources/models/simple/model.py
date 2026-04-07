from typing import Optional, Any

from djl_python import Input
from djl_python import Output

import logging

def postprocess(inputs: Input) -> Output:
    return Output()


def handle(inputs: Input) -> Optional[Output]:
    logging.info("Here Here .......")
    logging.info(inputs.get_data('name'))
    logging.info(inputs)

    outputs = Output()
    outputs.add(inputs.get_data('name'), "Name")
    outputs.add([1,6], "arr")
    outputs.add_property("message", "Welcome Java")
    return outputs