// Photon iMessage bridge: desk_assistant.py -> POST http://127.0.0.1:8787/send {to, text}
// Credentials come from ../.env (PHOTON_PROJECT_ID / PHOTON_PROJECT_SECRET). Messages are
// sent from the Photon project's iMessage line, not from your own phone number.
import { readFileSync, existsSync } from "node:fs";
import http from "node:http";
import { Spectrum } from "spectrum-ts";
import { imessage } from "@spectrum-ts/imessage";

const PORT = 8787;
const envPath = new URL("../.env", import.meta.url);
if (existsSync(envPath)) {
  for (const line of readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
    if (m && m[2] && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
  }
}
const { PHOTON_PROJECT_ID: projectId, PHOTON_PROJECT_SECRET: projectSecret } = process.env;
if (!projectId || !projectSecret) {
  console.error("PHOTON_PROJECT_ID / PHOTON_PROJECT_SECRET missing from .env");
  process.exit(1);
}

const app = await Spectrum({ projectId, projectSecret, providers: [imessage.config()] });
const im = imessage(app);
console.log("Photon connected");

async function send(to, text) {
  const space = await im.space.create(to);
  await space.send(text);
}

http
  .createServer((req, res) => {
    const reply = (code, obj) => {
      res.writeHead(code, { "Content-Type": "application/json" });
      res.end(JSON.stringify(obj));
    };
    if (req.method === "GET" && req.url === "/health") return reply(200, { ok: true });
    if (req.method !== "POST" || req.url !== "/send") return reply(404, { ok: false, error: "not found" });
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", async () => {
      try {
        const { to, text } = JSON.parse(body || "{}");
        if (!to || !text) return reply(400, { ok: false, error: "need 'to' and 'text'" });
        await send(to, text);
        console.log(`sent to ${to}: ${text}`);
        reply(200, { ok: true });
      } catch (e) {
        console.error("send failed:", e);
        reply(500, { ok: false, error: String(e?.message ?? e) });
      }
    });
  })
  .listen(PORT, "127.0.0.1", () => console.log(`Photon bridge on http://127.0.0.1:${PORT}`));
