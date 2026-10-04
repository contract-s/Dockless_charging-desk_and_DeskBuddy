# Shim: ESPHome >= 2026.9 excludes the ESP-IDF `bt` component unless something
# calls request_bluetooth(). wonderslug/esphome-ancs (pinned to 2026.5.3) doesn't,
# so its #include <nimble/nimble_port.h> fails. This re-includes `bt`.
import esphome.config_validation as cv
from esphome.components.esp32 import request_bluetooth

DEPENDENCIES = ["esp32"]

CONFIG_SCHEMA = cv.Schema({})


async def to_code(config):
    request_bluetooth()
