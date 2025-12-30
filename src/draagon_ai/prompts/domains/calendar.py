"""Calendar domain prompts.

Event creation, parsing, and management prompts.
"""

EVENT_CREATION_PROMPT = """Parse this natural language event request into structured event data.

TODAY'S DATE: {today}
USER REQUEST: {request}

Parse the request and extract:
- summary: Event title/name
- start_date: Start date in YYYY-MM-DD format
- start_time: Start time in HH:MM (24-hour) format, or null for all-day
- end_time: End time in HH:MM format (optional, default 1 hour after start)
- location: Event location (if mentioned)
- description: Any additional details
- all_day: true if no specific time mentioned

DATE PARSING RULES:
- "today" = {today}
- "tomorrow" = day after {today}
- "Saturday" or "this Saturday" = next occurrence of that day
- "next Saturday" = the Saturday after this week
- Specific dates like "December 15" or "12/15" = that date this year (or next year if passed)

TIME PARSING RULES:
- "7pm" = 19:00
- "3:30pm" = 15:30
- "noon" = 12:00
- "morning" = 09:00
- "evening" = 19:00
- "night" = 20:00
- If no time given, assume all-day event

Output XML:
<event>
    <summary>event title</summary>
    <start_date>YYYY-MM-DD</start_date>
    <start_time>HH:MM or empty</start_time>
    <end_time>HH:MM or empty</end_time>
    <location>location or empty</location>
    <description>details or empty</description>
    <all_day>true or false</all_day>
</event>

Examples:
- "Taylor Swift concert Saturday at 7pm" -> <event><summary>Taylor Swift concert</summary><start_date>2025-12-06</start_date><start_time>19:00</start_time><end_time>22:00</end_time><all_day>false</all_day></event>
- "Mom's birthday December 15" -> <event><summary>Mom's birthday</summary><start_date>2025-12-15</start_date><all_day>true</all_day></event>"""


EVENT_DETAILS_PROMPT = """Extract detailed event information from search results.

TODAY'S DATE: {today}
EVENT NAME: {event_name}
SEARCH RESULTS:
{search_results}

Extract as much detail as possible:
- full_name: Complete official name of the event
- date: Date in YYYY-MM-DD format (use {today}'s year unless specified otherwise)
- start_time: Start time in HH:MM (24-hour), or null if not found
- end_time: End time in HH:MM, or null if not found
- venue_name: Name of the venue
- venue_address: Full street address if available
- ticket_url: URL to buy tickets or event page
- description: Brief summary of the event (1-2 sentences)
- price: Ticket price or price range if mentioned

If information is not found in the search results, use null.
If multiple dates are mentioned, pick the one closest to today that hasn't passed.

Output XML:
<event_details>
    <full_name>event name</full_name>
    <date>YYYY-MM-DD</date>
    <start_time>HH:MM or empty</start_time>
    <end_time>HH:MM or empty</end_time>
    <venue_name>venue</venue_name>
    <venue_address>address or empty</venue_address>
    <ticket_url>url or empty</ticket_url>
    <description>summary</description>
    <price>price or empty</price>
</event_details>"""


CALENDAR_PROMPTS = {
    "EVENT_CREATION_PROMPT": EVENT_CREATION_PROMPT,
    "EVENT_DETAILS_PROMPT": EVENT_DETAILS_PROMPT,
}
