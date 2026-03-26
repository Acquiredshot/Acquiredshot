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
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate_fallback_summary(repo_summaries, lang_stats):
    """Generate a summary without an LLM (fallback)."""
    now = datetime.now(timezone.utc).strftime("%B %d, %Y")

    # Latest Innovation
    top_repos = repo_summaries[:3]
    repo_lines = []
    for r in top_repos:
        topics = ", ".join(r["topics"][:3]) if r["topics"] else r["language"]
        repo_lines.append(f"**{r['name']}** — {r['description']} `{topics}`")

    innovation = f"## 🔬 Latest Innovation\n\n"
    innovation += f"*Auto-generated on {now} by my Agentic Profile Bot* 🤖\n\n"
    innovation += "Most recent work:\n"
    for line in repo_lines:
        innovation += f"- {line}\n"

    # Skill DNA
    skill_dna = "\n## 🧬 Skill DNA — Current Focus\n\n"
    skill_dna += f"*Auto-generated on {now} by my Agentic Profile Bot* 🤖\n\n"
    for lang, pct in list(lang_stats.items())[:5]:
        filled = round(pct / 10)
        bar = "█" * filled + "░" * (10 - filled)
        skill_dna += f"**{lang}** {bar} {pct}%\n\n"

    return innovation + skill_dna


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


def main():
    print(f"🐺 Wolf-Pak Agentic Profile Updater")
    print(f"    Scanning repos for {GITHUB_USERNAME}...")

    repos = fetch_repos()
    print(f"    Found {len(repos)} repositories.")

    repo_summaries = build_repo_summary(repos)
    lang_stats = compute_language_stats(repos)
    print(f"    Language stats: {lang_stats}")

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
