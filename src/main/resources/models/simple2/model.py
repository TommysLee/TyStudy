from typing import Optional, Any

from djl_python import Input
from djl_python import Output

import time
import sys
import logging


def preprocess(inputs: Input):
    logging.info("################# Python Simple2 Model preprocess ... #################")
    time.sleep(2)
    logging.info("################# Python Simple2 Model preprocess completed #################")
    return None

def handle(inputs: Input) -> Optional[Output]:
    logging.info("################# Here Here !!! Python Simple2 Model Standby #################")
    logging.info("inputs = %s", inputs)

    name = ""
    if inputs.contains_key("name"):
        name = inputs.get_as_string("name")

    arr = [1, 6]
    logging.info("arr = %s", arr)

    outputs = Output()
    outputs.add(arr, "arr")
    outputs.add_property("message", "Hi, Deep Java Library From " + name)
    return outputs



def configLogging():
    logging.basicConfig(stream=sys.stdout,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        level=logging.INFO)

"""
    Python内测试
configLogging()

input = Input()
input.content.add("name", "Local")
output = handle(input)
logging.info("output = %s", output)
"""
