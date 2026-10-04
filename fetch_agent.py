"""
fetch_agent.py  —  the smart desk as a Fetch.ai agent you can talk to from ASI:One.

    ASI:One chat  ->  this agent (Agent Chat Protocol, Agentverse mailbox)
                  ->  desk_brain.py command server (127.0.0.1:8765)
                  ->  brain.py  ->  Spotify / gantry / focus timer / energy report
                  ->  reply back into the chat (and spoken on the desk)

Run (desk_brain.py must be running in another terminal):
    python fetch_agent.py
First run: open the "Agent inspector" link it prints, click Connect -> Mailbox.
Then in ASI:One (https://asi1.ai) ask e.g. "@smart-charging-desk play some lofi" or
"is my phone charging on my smart desk?". Copy the printed agent address into the README.

Needs Python 3.10+ and: pip install uagents requests
"""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import requests
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

import config

DESK_URL = f"http://{config.COMMAND_SERVER[0]}:{config.COMMAND_SERVER[1]}"

if not config.AGENT_SEED:
    raise SystemExit("Set AGENT_SEED in .env (any long random phrase; it is the agent's identity)")

agent = Agent(
    name=config.AGENT_NAME,
    seed=config.AGENT_SEED,
    port=config.AGENT_PORT,
    mailbox=True,
    publish_agent_details=True,
    readme_path="AGENT_README.md",
    description=("Smart charging desk: a camera finds your phone and a motorised wireless charger "
                 "slides under it. Controls the desk owner's Spotify music, the charger, a focus "
                 "timer, and reports phone charging status and energy saved."),
)

chat = Protocol(spec=chat_protocol_spec)


def ask_desk(text):
    try:
        r = requests.post(DESK_URL + "/command", json={"text": text}, timeout=20)
        r.raise_for_status()
        return r.json().get("reply") or "Done."
    except requests.ConnectionError:
        return "The desk is offline right now (desk_brain.py isn't running)."
    except Exception as e:
        return f"The desk hit an error: {e}"


@chat.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(timestamp=datetime.now(timezone.utc),
                                               acknowledged_msg_id=msg.msg_id))
    text = " ".join(c.text for c in msg.content if isinstance(c, TextContent)).strip()
    if not text:
        return
    ctx.logger.info(f"from {sender[:16]}...: {text}")
    reply = await asyncio.to_thread(ask_desk, text)
    ctx.logger.info(f"reply: {reply}")
    await ctx.send(sender, ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=reply), EndSessionContent(type="end-session")],
    ))


@chat.on_message(ChatAcknowledgement)
async def on_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(chat, publish_manifest=True)


@agent.on_event("startup")
async def hello(ctx: Context):
    ctx.logger.info(f"agent address: {agent.address}  (put this in the README)")
    try:
        s = await asyncio.to_thread(lambda: requests.get(DESK_URL + "/status", timeout=3).json())
        ctx.logger.info(f"desk is online: {s}")
    except Exception:
        ctx.logger.warning("desk_brain.py is not running yet; replies will say the desk is offline")


if __name__ == "__main__":
    agent.run()
