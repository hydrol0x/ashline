// Cloudflare Worker that passes Bot API calls on to Telegram, for networks that block api.telegram.org.
// Deploy: Cloudflare dashboard -> Workers & Pages -> Create -> Hello World -> paste this -> Deploy.
// Then in .env:  TELEGRAM_API_BASE=https://<your-worker>.workers.dev
// Only bot and file paths are forwarded; the bot token in the path is still what authorises each call.
export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (!/^\/(file\/)?bot\d+:[\w-]+\//.test(url.pathname)) {
      return new Response("Not found", { status: 404 });
    }
    const target = "https://api.telegram.org" + url.pathname + url.search;
    return fetch(new Request(target, request));
  },
};
