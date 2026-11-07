"""Command line entry point: starts both MCP servers, runs the graph, handles the review pause.

    python -m outreach_agent --company "Stripe" --role "Backend Engineer"
    python -m outreach_agent --resume stripe-backend-engineer-20261107-1532   # continue a paused review
    python -m outreach_agent --graph
"""

import argparse
import asyncio
import re
import sys
from datetime import datetime

from langgraph.types import Command

from .clients import AgentError
from .config import Settings
from .domains import slug
from .graph import build_graph
from .runtime import open_services


def new_run_id(company: str, role: str) -> str:
    return f"{slug(company)}-{slug(role)}-{datetime.now():%Y%m%d-%H%M%S}"


def show_shortlist(info: dict) -> None:
    print(f"\nSuggested people at {info['domain']}:")
    for i, p in enumerate(info["shortlist"], 1):
        email = f"{p.get('email')} ({p.get('email_confidence')}%)" if p.get("email") else "no email"
        print(f"  {i}. {p['name']}, {p.get('position')} [{p['category'].replace('_', ' ')}]")
        print(f"     {email} | {p.get('linkedin') or 'no LinkedIn'} | why: {p['why']}")


def parse_selection(raw: str, count: int) -> list[int]:
    raw = raw.strip().lower()
    if raw in {"all", "a", ""}:
        return list(range(count))
    if raw in {"none", "n", "q"}:
        return []
    return [int(x) - 1 for x in re.findall(r"\d+", raw) if 0 < int(x) <= count]


async def run(args: argparse.Namespace) -> None:
    settings = Settings()
    async with open_services(settings) as svc:
        app = build_graph(svc.people, svc.research, svc.llm, settings, svc.checkpointer)

        if args.resume:
            thread = args.resume
            snapshot = await app.aget_state({"configurable": {"thread_id": thread}})
            if not snapshot.tasks or not snapshot.tasks[0].interrupts:
                raise AgentError(f"No paused run with id {thread}.")
            result = {"__interrupt__": snapshot.tasks[0].interrupts}
        else:
            company = args.company or input("Company: ").strip()
            role = args.role or input("Role: ").strip()
            thread = new_run_id(company, role)
            inputs = {"company": company, "role": role, "domain": args.domain or "", "job_url": args.job_url or ""}
            result = await app.ainvoke(inputs, {"configurable": {"thread_id": thread}})
        config = {"configurable": {"thread_id": thread}}

        while "__interrupt__" in result:
            info = result["__interrupt__"][0].value
            show_shortlist(info)
            print(f"\n(Paused run id: {thread}. Resume later with --resume {thread})")
            raw = input("Who should I research and draft for? (e.g. 1,3,4 / all / none): ")
            result = await app.ainvoke(Command(resume=parse_selection(raw, len(info["shortlist"]))), config)

        if result.get("report_path"):
            print(f"\nDone. Open {result['report_path']} to review, edit and send the messages yourself.")
        elif not result.get("shortlist"):
            print("\nNo suitable recruiters or engineering managers found. Try --domain, or a bigger company.")
        else:
            print("\nNothing selected, nothing drafted.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare personalized cold outreach for a job.")
    parser.add_argument("--company")
    parser.add_argument("--role")
    parser.add_argument("--domain", help="company email domain, e.g. stripe.com (skips the lookup)")
    parser.add_argument("--job-url", help="link to the job posting (optional, improves the messages)")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue a run paused at the review step")
    parser.add_argument("--graph", action="store_true", help="print the graph as a Mermaid diagram")
    args = parser.parse_args()

    if args.graph:
        print(build_graph(None, None, None, Settings()).get_graph().draw_mermaid())
        return
    try:
        asyncio.run(run(args))
    except AgentError as e:
        sys.exit(f"Error: {e}")
    except KeyboardInterrupt:
        sys.exit("\nStopped.")
