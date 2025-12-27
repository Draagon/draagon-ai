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

Output JSON:
{{"summary":"event title","start_date":"YYYY-MM-DD","start_time":"HH:MM or null","end_time":"HH:MM or null","location":"location or null","description":"details or null","all_day":true/false}}

Examples:
- "Taylor Swift concert Saturday at 7pm" -> {{"summary":"Taylor Swift concert","start_date":"2025-12-06","start_time":"19:00","end_time":"22:00","location":null,"description":null,"all_day":false}}
- "Dentist appointment tomorrow at 2:30pm" -> {{"summary":"Dentist appointment","start_date":"2025-12-06","start_time":"14:30","end_time":"15:30","location":null,"description":null,"all_day":false}}
- "Mom's birthday December 15" -> {{"summary":"Mom's birthday","start_date":"2025-12-15","start_time":null,"end_time":null,"location":null,"description":null,"all_day":true}}"""


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

Output JSON:
{{"full_name":"event name","date":"YYYY-MM-DD","start_time":"HH:MM or null","end_time":"HH:MM or null","venue_name":"venue","venue_address":"address or null","ticket_url":"url or null","description":"summary","price":"price or null"}}"""


CALENDAR_PROMPTS = {
    "EVENT_CREATION_PROMPT": EVENT_CREATION_PROMPT,
    "EVENT_DETAILS_PROMPT": EVENT_DETAILS_PROMPT,
}
