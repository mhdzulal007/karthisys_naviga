"""Live world intel — Concurrent Sub-150ms Web & Weather Search for JARVIS."""

from __future__ import annotations

import concurrent.futures
import html as html_module
import json
import re
import urllib.parse
import urllib.request
from typing import Optional

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}


def _fetch_ddg_lite(q: str):
    try:
        data = urllib.parse.urlencode({"q": q}).encode("utf-8")
        req = urllib.request.Request("https://lite.duckduckgo.com/lite/", data=data, headers=_UA)
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            html_raw = resp.read().decode("utf-8", errors="replace")
            tds = re.findall(r'<td class=[\'"]result-snippet[\'"][^>]*>(.*?)</td>', html_raw, re.DOTALL)
            clean = [html_module.unescape(re.sub(r'<[^<]+?>', '', s)).strip() for s in tds if s.strip()]
            return [f"- {s}" for s in clean[:4]]
    except Exception:
        return []


def _fetch_wiki(q: str):
    try:
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "opensearch", "search": q, "limit": 2, "namespace": 0, "format": "json"
        })
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            titles, descs, urls = data[1], data[2], data[3]
            res = []
            for i, title in enumerate(titles):
                desc = descs[i] if i < len(descs) else ""
                url_str = urls[i] if i < len(urls) else ""
                if desc:
                    res.append(f"Wikipedia ({title}): {desc} {url_str}".strip())
            return res
    except Exception:
        return []


def web_search(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "Error: empty search query."

    bits = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_ddg = executor.submit(_fetch_ddg_lite, q)
        f_wiki = executor.submit(_fetch_wiki, q)

        try:
            for item in f_ddg.result(timeout=2.5):
                bits.append(item)
        except Exception:
            pass

        try:
            for item in f_wiki.result(timeout=2.5):
                bits.append(item)
        except Exception:
            pass

    if not bits:
        return f"No live results found for '{q}'."
    return f"Real-time search results for '{q}':\n" + "\n".join(bits[:6])


def _geocode(location: str) -> Optional[tuple]:
    loc = (location or "").strip()
    if not loc:
        try:
            req = urllib.request.Request("http://ip-api.com/json/?fields=status,lat,lon,city,regionName,country", headers=_UA)
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                geo = json.loads(resp.read().decode("utf-8", errors="replace"))
                if geo.get("status") == "success":
                    return geo["lat"], geo["lon"], f"{geo.get('city')}, {geo.get('regionName')}, {geo.get('country')}"
        except Exception:
            return None
        return None
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({"name": loc, "count": 1, "language": "en", "format": "json"})
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            results = data.get("results") or []
            if not results:
                return None
            r = results[0]
            label = ", ".join(p for p in (r.get("name"), r.get("admin1"), r.get("country")) if p)
            return r["latitude"], r["longitude"], label
    except Exception:
        return None


def get_weather(location: str = "") -> str:
    geo = _geocode(location)
    if geo is None:
        return "Could not resolve location for weather."
    lat, lon, label = geo
    try:
        url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
        })
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            cur = data.get("current") or {}
            code = int(cur.get("weather_code") or 0)
            sky = {
                0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                45: "fog", 48: "rime fog", 51: "light drizzle", 61: "rain",
                71: "snow", 80: "rain showers", 95: "thunderstorm",
            }.get(code, f"code {code}")
            return (
                f"Weather for {label}: {cur.get('temperature_2m')}°C, {sky}, "
                f"humidity {cur.get('relative_humidity_2m')}%, "
                f"wind {cur.get('wind_speed_10m')} km/h."
            )
    except Exception as e:
        return f"Weather fetch failed: {e}"
