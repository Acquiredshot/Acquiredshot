"""
Agentic Profile Updater
Scans Acquiredshot's repositories, analyzes tech stacks and recent activity,
then uses an LLM to generate a dynamic "Latest Innovation" summary and
"Skill DNA" breakdown for the profile README.
"""

import json
import os
import re
from datetime import datetime, timezone

import requests

GITHUB_USERNAME = "Acquiredshot"
GITHUB_TOKEN = os.environ.get("GH_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
README_PATH = "README.md"

HEADERS_GH = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}


def fetch_repos():
    """Fetch all public repos for the user."""
    repos = []
    page = 1
    while True:
        resp = requests.get(
            f"https://api.github.com/users/{GITHUB_USERNAME}/repos",
            headers=HEADERS_GH,
            params={"per_page": 100, "page": page, "sort": "updated", "type": "public"},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        # SECURITY: Only include public, non-fork repos — never expose private repo data
        public_only = [r for r in batch if not r.get("private") and not r.get("fork")]
        repos.extend(public_only)
        page += 1
    return repos


def fetch_recent_commits(repo_name, count=10):
    """Fetch the most recent commits for a public repo. Only returns commit messages (no diffs/code)."""
    resp = requests.get(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/commits",
        headers=HEADERS_GH,
        params={"per_page": count},
        timeout=30,
    )
    if resp.status_code != 200:
        return []
    commits = resp.json()
    # SECURITY: Strip to only commit messages — never send code, patches, or file contents
    return [
        {"commit": {"message": c.get("commit", {}).get("message", "")}}
        for c in commits
        if isinstance(c, dict)
    ]


def build_repo_summary(repos):
    """Build a structured summary of repos and their tech stacks."""
    summaries = []
    for repo in repos[:20]:  # Top 20 most recently updated
        # SECURITY: Double-check — skip any repo that isn't public
        if repo.get("private"):
            continue
        info = {
            "name": repo["name"],
            "description": repo.get("description") or "No description",
            "language": repo.get("language") or "Unknown",
            "stars": repo.get("stargazers_count", 0),
            "forks": repo.get("forks_count", 0),
            "updated": repo.get("updated_at", ""),
            "topics": repo.get("topics", []),
        }

        # Fetch recent commit messages for context
        commits = fetch_recent_commits(repo["name"], count=5)
        info["recent_commits"] = [
            c.get("commit", {}).get("message", "").split("\n")[0]
            for c in commits
            if isinstance(c, dict)
        ][:5]

        summaries.append(info)
    return summaries


def compute_language_stats(repos):
    """Compute language distribution across all repos."""
    lang_counts = {}
    for repo in repos:
        lang = repo.get("language")
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    total = sum(lang_counts.values())
    if total == 0:
        return {}
    return {
        lang: round((count / total) * 100)
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])
    }


def generate_ai_summary(repo_summaries, lang_stats):
    """Use OpenAI to generate the profile sections."""
    if not OPENAI_API_KEY:
        return generate_fallback_summary(repo_summaries, lang_stats)

    prompt = f"""You are an AI assistant that writes GitHub profile README sections.
Given the following data about a developer named Cody (GitHub: Acquiredshot), generate two Markdown sections.

## Data
**Repository Summaries (most recently updated):**
{json.dumps(repo_summaries, indent=2)}

**Language Distribution:**
{json.dumps(lang_stats, indent=2)}

**Current Date:** {datetime.now(timezone.utc).strftime('%B %d, %Y')}

## Instructions
Generate EXACTLY two sections in Markdown. No extra commentary.

### Section 1: "🔬 Latest Innovation"
- Write 2-3 sentences summarizing the most interesting/recent work across the repos.
- Highlight specific project names, tech stacks, and what makes them notable.
- Tone: professional but energetic. Show this developer is actively building.

### Section 2: "🧬 Skill DNA — This Week's Focus"
- Create a visual breakdown using text-based progress bars.
- Analyze the recent commit messages and repo activity to determine focus areas.
- Format each line like: `**Category Name** ████████░░ 80%`
- Use filled █ and empty ░ blocks (10 total) to represent percentages.
- Show top 4-5 skill categories.
- Categories should be specific (e.g., "Backend Security Architecture", "Agentic AI Logic", "Frontend React Development") not generic.

Output ONLY the two sections in valid Markdown. Start with the ## heading for each."""

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7,
            },
            timeout=60,
        )
        resp.raise_for_status()
        # Always use animated SVGs in the README, ignore LLM markdown output
        # (The LLM analysis is for future use; SVGs look way better)
        return generate_svg_markdown()
    except Exception as e:
        print(f"    ⚠️ OpenAI API error: {e}")
        print("    Falling back to stats-based summary...")
        return generate_fallback_summary(repo_summaries, lang_stats)


def generate_fallback_summary(repo_summaries, lang_stats):
    """Generate a summary without an LLM (fallback) — now uses animated SVGs."""
    return generate_svg_markdown()


def generate_svg_markdown():
    """Return markdown that embeds the animated SVGs."""
    return (
        '<div align="center">\n'
        '  <img src="https://raw.githubusercontent.com/Acquiredshot/Acquiredshot/main/assets/latest-innovation.svg" width="100%" alt="Latest Innovation"/>\n'
        '</div>\n\n'
        '<div align="center">\n'
        '  <img src="https://raw.githubusercontent.com/Acquiredshot/Acquiredshot/main/assets/skill-dna.svg" width="100%" alt="Skill DNA"/>\n'
        '</div>'
    )


def update_readme(ai_content):
    """Replace the AI-generated section in the README."""
    with open(README_PATH, "r", encoding="utf-8") as f:
        readme = f.read()

    start_marker = "<!--START_SECTION:ai-profile-->"
    end_marker = "<!--END_SECTION:ai-profile-->"

    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker),
        re.DOTALL,
    )

    new_section = f"{start_marker}\n{ai_content}\n{end_marker}"

    if pattern.search(readme):
        updated = pattern.sub(new_section, readme)
    else:
        print("WARNING: AI profile markers not found in README. No update performed.")
        return False

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated)

    return True


# ──────────────────────────────────────────────
#  Animated SVG Generators
# ──────────────────────────────────────────────

LANG_COLORS = {
    "Python": "#3572A5",
    "Java": "#b07219",
    "TypeScript": "#3178c6",
    "JavaScript": "#f1e05a",
    "Go": "#00ADD8",
    "CSS": "#563d7c",
    "HTML": "#e34c26",
    "C": "#555555",
    "C++": "#f34b7d",
    "C#": "#178600",
    "Ruby": "#701516",
    "Rust": "#dea584",
    "Swift": "#F05138",
    "Kotlin": "#A97BFF",
    "Shell": "#89e051",
    "Dart": "#00B4AB",
    "PHP": "#4F5D95",
    "Scala": "#c22d40",
    "R": "#198CE7",
    "Lua": "#000080",
}


def _escape_xml(text):
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def generate_innovation_svg(repo_summaries):
    """Generate an animated SVG card for Latest Innovation."""
    now = datetime.now(timezone.utc).strftime("%B %d, %Y")
    top_repos = repo_summaries[:4]

    card_height = 220 + len(top_repos) * 85
    repo_blocks = ""

    for i, repo in enumerate(top_repos):
        y_offset = 180 + i * 85
        delay = 0.5 + i * 0.3
        name = _escape_xml(repo["name"])
        desc = _escape_xml(repo.get("description", "No description")[:80])
        lang = repo.get("language", "Unknown")
        lang_color = LANG_COLORS.get(lang, "#70a5fd")
        stars = repo.get("stargazers_count", 0)
        topics = repo.get("topics", [])[:3]
        # Dynamically size topic badges based on text length
        tag_parts = []
        tag_x = 305
        for t in topics:
            label = _escape_xml(t)
            char_w = 5.5  # approx width per char at font-size 8
            pad = 14  # horizontal padding
            w = max(len(t) * char_w + pad, 40)
            tag_parts.append(
                f'<text x="{tag_x}" y="{y_offset + 55}" font-size="8" fill="#c0caf5" font-family="Segoe UI, Ubuntu, sans-serif">{label}</text>'
            )
            tag_x += w + 8
        topic_tags = "".join(tag_parts)

        repo_blocks += f"""
    <g opacity="0">
      <animate attributeName="opacity" from="0" to="1" dur="0.6s" begin="{delay}s" fill="freeze"/>
      <rect x="30" y="{y_offset}" width="740" height="70" rx="12" fill="#1a1b27" stroke="#38bdae" stroke-width="1" opacity="0.6">
        <animate attributeName="stroke-opacity" values="0.3;0.8;0.3" dur="3s" repeatCount="indefinite" begin="{delay}s"/>
      </rect>
      <circle cx="55" cy="{y_offset + 20}" r="6" fill="{lang_color}">
        <animate attributeName="r" values="5;7;5" dur="2s" repeatCount="indefinite"/>
      </circle>
      <text x="70" y="{y_offset + 24}" font-size="15" font-weight="bold" fill="#70a5fd" font-family="Segoe UI, Ubuntu, sans-serif">{name}</text>
      <text x="70" y="{y_offset + 46}" font-size="12" fill="#a9b1d6" font-family="Segoe UI, Ubuntu, sans-serif">{desc}</text>
      <text x="70" y="{y_offset + 62}" font-size="11" fill="{lang_color}" font-family="Segoe UI, Ubuntu, sans-serif">{_escape_xml(lang)}</text>
      <text x="160" y="{y_offset + 62}" font-size="11" fill="#c0caf5" font-family="Segoe UI, Ubuntu, sans-serif">{stars}</text>
      {topic_tags}
    </g>"""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="800" height="{card_height}" viewBox="0 0 800 {card_height}">
  <defs>
    <linearGradient id="headerGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#a9fef7"/>
      <stop offset="50%" style="stop-color:#bf91f3"/>
      <stop offset="100%" style="stop-color:#70a5fd"/>
      <animate attributeName="x1" values="0%;100%;0%" dur="6s" repeatCount="indefinite"/>
      <animate attributeName="x2" values="100%;200%;100%" dur="6s" repeatCount="indefinite"/>
    </linearGradient>
    <linearGradient id="borderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#a9fef7;stop-opacity:0.6"/>
      <stop offset="50%" style="stop-color:#bf91f3;stop-opacity:0.6"/>
      <stop offset="100%" style="stop-color:#70a5fd;stop-opacity:0.6"/>
      <animate attributeName="x1" values="0%;100%;0%" dur="4s" repeatCount="indefinite"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <linearGradient id="scanLine" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#a9fef7;stop-opacity:0"/>
      <stop offset="50%" style="stop-color:#a9fef7;stop-opacity:0.08"/>
      <stop offset="100%" style="stop-color:#a9fef7;stop-opacity:0"/>
    </linearGradient>
  </defs>

  <rect width="800" height="{card_height}" rx="16" fill="#0d1117"/>
  <rect width="800" height="{card_height}" rx="16" fill="none" stroke="url(#borderGrad)" stroke-width="2"/>

  <rect x="0" y="0" width="800" height="80" fill="url(#scanLine)" rx="16">
    <animateTransform attributeName="transform" type="translate" values="0,-80;0,{card_height}" dur="4s" repeatCount="indefinite"/>
  </rect>

  <g opacity="0">
    <animate attributeName="opacity" from="0" to="1" dur="0.8s" begin="0.1s" fill="freeze"/>
    <text x="50" y="55" font-size="26" font-weight="bold" fill="url(#headerGrad)" font-family="Segoe UI, Ubuntu, sans-serif" filter="url(#glow)">Latest Innovation</text>
  </g>

  <line x1="50" y1="70" x2="50" y2="70" stroke="url(#headerGrad)" stroke-width="2" stroke-linecap="round">
    <animate attributeName="x2" from="50" to="350" dur="1s" begin="0.3s" fill="freeze"/>
  </line>

  <g opacity="0">
    <animate attributeName="opacity" from="0" to="0.6" dur="0.5s" begin="0.4s" fill="freeze"/>
    <text x="50" y="100" font-size="11" fill="#a9fef7" font-family="Segoe UI, Ubuntu, sans-serif">Auto-generated on {now} by Wolf-Pak Agentic Profile Bot</text>
  </g>

  <circle cx="720" cy="40" r="2" fill="#a9fef7" opacity="0.5">
    <animate attributeName="cy" values="40;30;40" dur="3s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.3;0.8;0.3" dur="3s" repeatCount="indefinite"/>
  </circle>
  <circle cx="740" cy="55" r="1.5" fill="#bf91f3" opacity="0.5">
    <animate attributeName="cy" values="55;45;55" dur="2.5s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.3;0.7;0.3" dur="2.5s" repeatCount="indefinite"/>
  </circle>
  <circle cx="760" cy="35" r="1" fill="#70a5fd" opacity="0.5">
    <animate attributeName="cy" values="35;25;35" dur="2s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.2;0.9;0.2" dur="2s" repeatCount="indefinite"/>
  </circle>

  <g opacity="0">
    <animate attributeName="opacity" from="0" to="1" dur="0.5s" begin="0.5s" fill="freeze"/>
    <text x="50" y="145" font-size="14" font-weight="bold" fill="#c0caf5" font-family="Segoe UI, Ubuntu, sans-serif">Most Active Projects</text>
    <line x1="50" y1="155" x2="200" y2="155" stroke="#38bdae" stroke-width="1" opacity="0.4"/>
  </g>

  {repo_blocks}
</svg>"""
    return svg


def generate_skill_dna_svg(lang_stats):
    """Generate an animated SVG card for Skill DNA."""
    now = datetime.now(timezone.utc).strftime("%B %d, %Y")
    langs = list(lang_stats.items())[:6]

    card_height = 200 + len(langs) * 65
    bar_blocks = ""

    bar_gradients = [
        ("#a9fef7", "#38bdae"),
        ("#bf91f3", "#7c3aed"),
        ("#70a5fd", "#3b82f6"),
        ("#f7c177", "#f59e0b"),
        ("#ff7eb3", "#ec4899"),
        ("#89e051", "#22c55e"),
    ]

    gradient_defs = ""
    for i, (color1, color2) in enumerate(bar_gradients):
        gradient_defs += f"""
    <linearGradient id="bar{i}" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:{color1}"/>
      <stop offset="100%" style="stop-color:{color2}"/>
    </linearGradient>"""

    for i, (lang, pct) in enumerate(langs):
        y_offset = 160 + i * 65
        delay = 0.4 + i * 0.25
        bar_width = max(int((pct / 100) * 550), 20)
        lang_color = LANG_COLORS.get(lang, "#70a5fd")
        grad_id = f"bar{i % len(bar_gradients)}"

        bar_blocks += f"""
    <g opacity="0">
      <animate attributeName="opacity" from="0" to="1" dur="0.5s" begin="{delay}s" fill="freeze"/>
      <circle cx="55" cy="{y_offset + 14}" r="6" fill="{lang_color}">
        <animate attributeName="r" values="5;7;5" dur="2.5s" repeatCount="indefinite" begin="{delay}s"/>
      </circle>
      <text x="70" y="{y_offset + 18}" font-size="14" font-weight="bold" fill="#c0caf5" font-family="Segoe UI, Ubuntu, sans-serif">{_escape_xml(lang)}</text>
      <text x="740" y="{y_offset + 18}" font-size="14" font-weight="bold" fill="url(#{grad_id})" font-family="Segoe UI, Ubuntu, sans-serif" text-anchor="end">{pct}%</text>
      <rect x="55" y="{y_offset + 28}" width="550" height="16" rx="8" fill="#1a1b27" stroke="#282d3f" stroke-width="1"/>
      <rect x="55" y="{y_offset + 28}" width="0" height="16" rx="8" fill="url(#{grad_id})">
        <animate attributeName="width" from="0" to="{bar_width}" dur="1.2s" begin="{delay}s" fill="freeze" calcMode="spline" keySplines="0.25 0.1 0.25 1"/>
      </rect>
      <rect x="55" y="{y_offset + 28}" width="0" height="16" rx="8" fill="url(#{grad_id})" opacity="0.3" filter="url(#glow)">
        <animate attributeName="width" from="0" to="{bar_width}" dur="1.2s" begin="{delay}s" fill="freeze"/>
        <animate attributeName="opacity" values="0.1;0.4;0.1" dur="3s" repeatCount="indefinite" begin="{delay + 1.2}s"/>
      </rect>
    </g>"""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="800" height="{card_height}" viewBox="0 0 800 {card_height}">
  <defs>
    <linearGradient id="dnaHeaderGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#a9fef7"/>
      <stop offset="33%" style="stop-color:#bf91f3"/>
      <stop offset="66%" style="stop-color:#70a5fd"/>
      <stop offset="100%" style="stop-color:#a9fef7"/>
      <animate attributeName="x1" values="-100%;0%;-100%" dur="5s" repeatCount="indefinite"/>
      <animate attributeName="x2" values="0%;100%;0%" dur="5s" repeatCount="indefinite"/>
    </linearGradient>
    <linearGradient id="dnaBorder" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#bf91f3;stop-opacity:0.5"/>
      <stop offset="100%" style="stop-color:#a9fef7;stop-opacity:0.5"/>
      <animate attributeName="x1" values="0%;100%;0%" dur="6s" repeatCount="indefinite"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    {gradient_defs}
    <linearGradient id="dnaScan" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#bf91f3;stop-opacity:0"/>
      <stop offset="50%" style="stop-color:#bf91f3;stop-opacity:0.06"/>
      <stop offset="100%" style="stop-color:#bf91f3;stop-opacity:0"/>
    </linearGradient>
  </defs>

  <rect width="800" height="{card_height}" rx="16" fill="#0d1117"/>
  <rect width="800" height="{card_height}" rx="16" fill="none" stroke="url(#dnaBorder)" stroke-width="2"/>

  <rect x="0" y="0" width="800" height="60" fill="url(#dnaScan)" rx="16">
    <animateTransform attributeName="transform" type="translate" values="0,-60;0,{card_height}" dur="5s" repeatCount="indefinite"/>
  </rect>

  <!-- DNA Helix decoration -->
  <g opacity="0.15">
    <circle cx="760" cy="80" r="4" fill="#a9fef7">
      <animate attributeName="cx" values="755;765;755" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="740" cy="80" r="4" fill="#bf91f3">
      <animate attributeName="cx" values="745;735;745" dur="2s" repeatCount="indefinite"/>
    </circle>
    <line x1="740" y1="80" x2="760" y2="80" stroke="#70a5fd" stroke-width="1">
      <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2s" repeatCount="indefinite"/>
    </line>
    <circle cx="760" cy="120" r="4" fill="#bf91f3">
      <animate attributeName="cx" values="765;755;765" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="740" cy="120" r="4" fill="#a9fef7">
      <animate attributeName="cx" values="735;745;735" dur="2s" repeatCount="indefinite"/>
    </circle>
    <line x1="740" y1="120" x2="760" y2="120" stroke="#70a5fd" stroke-width="1">
      <animate attributeName="opacity" values="0.6;0.3;0.6" dur="2s" repeatCount="indefinite"/>
    </line>
    <circle cx="760" cy="160" r="4" fill="#a9fef7">
      <animate attributeName="cx" values="755;765;755" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="740" cy="160" r="4" fill="#bf91f3">
      <animate attributeName="cx" values="745;735;745" dur="2s" repeatCount="indefinite"/>
    </circle>
    <line x1="740" y1="160" x2="760" y2="160" stroke="#70a5fd" stroke-width="1">
      <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2s" repeatCount="indefinite"/>
    </line>
  </g>

  <g opacity="0">
    <animate attributeName="opacity" from="0" to="1" dur="0.8s" begin="0.1s" fill="freeze"/>
    <text x="50" y="55" font-size="26" font-weight="bold" fill="url(#dnaHeaderGrad)" font-family="Segoe UI, Ubuntu, sans-serif" filter="url(#glow)">Skill DNA — Current Focus</text>
  </g>

  <line x1="50" y1="70" x2="50" y2="70" stroke="url(#dnaHeaderGrad)" stroke-width="2" stroke-linecap="round">
    <animate attributeName="x2" from="50" to="420" dur="1s" begin="0.3s" fill="freeze"/>
  </line>

  <g opacity="0">
    <animate attributeName="opacity" from="0" to="0.6" dur="0.5s" begin="0.4s" fill="freeze"/>
    <text x="50" y="100" font-size="11" fill="#bf91f3" font-family="Segoe UI, Ubuntu, sans-serif">Auto-generated on {now} by Wolf-Pak Agentic Profile Bot</text>
    <text x="50" y="120" font-size="11" fill="#565f89" font-family="Segoe UI, Ubuntu, sans-serif">Based on analysis of {sum(lang_stats.values())} repositories</text>
  </g>

  {bar_blocks}
</svg>"""
    return svg


def save_svgs(repo_summaries, lang_stats):
    """Generate and save animated SVGs."""
    os.makedirs("assets", exist_ok=True)

    innovation_svg = generate_innovation_svg(repo_summaries)
    with open("assets/latest-innovation.svg", "w", encoding="utf-8") as f:
        f.write(innovation_svg)
    print("    Generated assets/latest-innovation.svg")

    skill_svg = generate_skill_dna_svg(lang_stats)
    with open("assets/skill-dna.svg", "w", encoding="utf-8") as f:
        f.write(skill_svg)
    print("    Generated assets/skill-dna.svg")


def main():
    print(f"🐺 Wolf-Pak Agentic Profile Updater")
    print(f"    Scanning repos for {GITHUB_USERNAME}...")

    repos = fetch_repos()
    print(f"    Found {len(repos)} repositories.")

    repo_summaries = build_repo_summary(repos)
    lang_stats = compute_language_stats(repos)
    print(f"    Language stats: {lang_stats}")

    print("    Generating animated SVGs...")
    save_svgs(repo_summaries, lang_stats)

    print("    Generating AI summary...")
    ai_content = generate_ai_summary(repo_summaries, lang_stats)

    print("    Updating README...")
    success = update_readme(ai_content)

    if success:
        print("    ✅ README updated successfully!")
    else:
        print("    ❌ Failed to update README — markers not found.")


if __name__ == "__main__":
    main()
