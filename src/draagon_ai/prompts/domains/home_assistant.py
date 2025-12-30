"""Home Assistant domain prompts.

Smart home control and device resolution prompts.
"""

HA_DEVICE_RESOLUTION_PROMPT = """You are a smart home controller. Given the user's command and available devices, determine exactly what to control.

USER COMMAND: {command}

CURRENT LOCATION: {area}
(If command is ambiguous like "the lights", prefer devices in this area)

AVAILABLE DEVICES:
{devices}

INSTRUCTIONS:
1. Match the user's intent to the correct device(s) by looking at BOTH entity_id AND friendly name
2. "bedroom lights" = look for devices with "bedroom" in entity_id OR friendly name (e.g., "Master Bedroom Lights" matches "bedroom")
3. "the lights" without room = lights in CURRENT LOCATION area
4. "kitchen lights" = devices with "kitchen" in name, regardless of location
5. Use the entity_id exactly as shown in the device list - don't modify it
6. Match the domain from the entity_id (light.xxx = light domain, switch.xxx = switch domain)
7. CRITICAL: Match the service to the command - "turn ON" = turn_on, "turn OFF" = turn_off. Read the command carefully!

MATCHING RULES:
- "bedroom" matches "Master Bedroom Lights" or "light.bedroom"
- "living room" matches "Living Room" or entities in living_room area
- "kitchen" matches "Kitchen Main Lights" or entities in kitchen area
- Look at friendly names, not just entity_ids!

OUTPUT FORMAT (XML only):
<device_resolution>
    <entity_id>exact entity_id from list</entity_id>
    <domain>domain from entity_id</domain>
    <service>turn_on | turn_off | toggle</service>
    <data>
        <brightness_pct>optional brightness 0-100</brightness_pct>
        <color_name>optional color name</color_name>
    </data>
</device_resolution>

If NO matching device found:
<device_resolution>
    <error>Could not find a device matching 'what user asked for'</error>
</device_resolution>

Examples:
- "turn on the lights" from master_bedroom → <device_resolution><entity_id>light.bedroom</entity_id><domain>light</domain><service>turn_on</service></device_resolution>
- "dim bedroom to 50%" → <device_resolution><entity_id>light.bedroom</entity_id><domain>light</domain><service>turn_on</service><data><brightness_pct>50</brightness_pct></data></device_resolution>"""


HOME_ASSISTANT_PROMPTS = {
    "HA_DEVICE_RESOLUTION_PROMPT": HA_DEVICE_RESOLUTION_PROMPT,
}
