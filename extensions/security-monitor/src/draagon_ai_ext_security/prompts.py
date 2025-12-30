"""Prompts for security analysis."""

CLASSIFICATION_PROMPT = """You are a security analyst for a HOME network (not enterprise).

NETWORK CONTEXT:
- HOME_NET: {home_net}
- Known services: {known_services}
- This is Doug's home network with typical home devices and services

ALERT DATA:
{alert_json}

PAST SIMILAR ISSUES:
{memory_context}

Common FALSE POSITIVES in home networks:
- ET POLICY: Software updates (apt, snap, pip, npm)
- ET INFO: Streaming services (Netflix, Plex, YouTube)
- ET SCAN: Known scanners (Shodan, Censys, academic research)
- ET DNS: Ad networks, analytics, CDNs (usually blocked by Pi-hole anyway)
- ET USER_AGENTS: Legitimate software with unusual user agents

Signs of REAL THREATS:
- Connections to known C2 domains/IPs
- Unusual outbound connections to foreign IPs
- Multiple failed auth attempts from single source
- Data exfiltration patterns (large outbound transfers)
- Lateral movement within HOME_NET

Classify this alert. Output XML:
<classification>
  <threat_level>critical|high|medium|low|noise</threat_level>
  <is_false_positive>true|false</is_false_positive>
  <confidence>0.0-1.0</confidence>
  <reasoning>Brief explanation of your classification</reasoning>
  <action_required>What Doug should do (or 'None')</action_required>
  <needs_investigation>true|false</needs_investigation>
</classification>
"""

INVESTIGATION_PROMPT = """You are investigating a security alert on Doug's home network.

ALERT DETAILS:
{alert}

NETWORK CONTEXT:
{network_context}

PAST INVESTIGATIONS:
{memory_context}

YOUR GOAL:
Determine if this is a real threat requiring action, or a false positive.

AVAILABLE TOOLS:
{tools}

INVESTIGATION APPROACH:
1. First, check memory for past encounters with this IP/domain/signature
2. If unknown, search for threat intelligence
3. Check if firewall/Pi-hole is already blocking
4. Correlate with other recent alerts
5. Determine confidence level and recommended action

Be thorough but efficient. Most home network alerts are noise, but don't dismiss
anything that could indicate real compromise.

After investigation, provide XML:
<investigation>
  <threat_level>critical|high|medium|low|noise</threat_level>
  <summary>One-paragraph summary of findings</summary>
  <detailed_findings>Full investigation details</detailed_findings>
  <action_required>Specific actions Doug should take</action_required>
  <should_notify>true|false</should_notify>
  <store_in_memory>true|false</store_in_memory>
</investigation>
"""

VOICE_ANNOUNCEMENT_PROMPT = """Convert this security finding into a natural voice announcement for Doug.

FINDING:
{finding_json}

RULES:
1. Keep under 30 seconds of speech (~75 words max)
2. Lead with severity context:
   - Critical: "Security alert."
   - High: "Security notice."
   - Medium/Low: "Quick security update."
3. Explain in plain English - avoid technical jargon
4. Don't read out IP addresses unless critical
5. Always end with action required or "no action needed"
6. Be conversational, like a helpful assistant

EXAMPLES:

Critical threat:
"Security alert. I detected suspicious connections to a known malware server from a device on your network. I've blocked the traffic, but you should scan your devices for malware as soon as possible."

High threat:
"Security notice. Someone attempted to log into your server 47 times in the last hour from an IP in Russia. Your firewall blocked all attempts, but you may want to check your fail2ban settings."

Medium (for summary):
"Quick security update. I noticed some port scanning from a known research scanner. Your firewall handled it. No action needed."

Output only the announcement text, nothing else.
"""

DAILY_SUMMARY_PROMPT = """Create a brief daily security summary for Doug.

STATS (last 24 hours):
{stats}

NOTABLE EVENTS:
{notable_events}

RULES:
1. Keep concise - this is for a quick morning briefing
2. Lead with overall status (all clear, some concerns, action needed)
3. Highlight anything requiring attention
4. Mention trends if interesting
5. End with confidence level

Format as natural speech for voice announcement (~45 seconds max).
"""

# =============================================================================
# PROGRESSIVE DISCOVERY PROMPTS
# =============================================================================

UNKNOWN_DEVICE_DISCOVERY_PROMPT = """An alert mentions an IP address that isn't in the known network services list.

ALERT CONTEXT:
{alert_context}

IP ADDRESS: {ip_address}
CURRENT KNOWN SERVICES:
{known_services}

Determine if this is:
1. A new device the user should tell us about (internal IP on home network)
2. An external/internet IP (no need to ask)
3. A known service by another name (fuzzy match)

Output XML:
<discovery>
  <is_internal>true|false</is_internal>
  <should_ask_user>true|false</should_ask_user>
  <possible_matches>
    <match>service name</match>
  </possible_matches>
  <suggested_question>Natural question to ask the user, or empty</suggested_question>
  <reasoning>Why we should/shouldn't ask</reasoning>
</discovery>

EXAMPLES:

IP 192.168.168.215 with DNS traffic:
<discovery>
  <is_internal>true</is_internal>
  <should_ask_user>true</should_ask_user>
  <possible_matches></possible_matches>
  <suggested_question>I noticed some traffic from 192.168.168.215. Is that a device I should know about?</suggested_question>
  <reasoning>Internal IP not in known services, appears to be a legitimate device</reasoning>
</discovery>

IP 8.8.8.8:
<discovery>
  <is_internal>false</is_internal>
  <should_ask_user>false</should_ask_user>
  <possible_matches></possible_matches>
  <suggested_question></suggested_question>
  <reasoning>Google DNS server, external public IP</reasoning>
</discovery>

IP 192.168.168.204 (when Plex is at .204):
<discovery>
  <is_internal>true</is_internal>
  <should_ask_user>false</should_ask_user>
  <possible_matches>
    <match>Plex</match>
  </possible_matches>
  <suggested_question></suggested_question>
  <reasoning>Matches known Plex server IP</reasoning>
</discovery>
"""

LEARN_SERVICE_FROM_RESPONSE_PROMPT = """The user just told us about a network service. Extract the details.

USER'S RESPONSE:
"{user_response}"

CONTEXT (what we asked about):
{context}

Extract the service information. Output XML:
<service>
  <name>Service name (e.g., 'NAS', 'Printer', 'Smart TV')</name>
  <ip>IP address mentioned or inferred</ip>
  <ports>
    <port>port number</port>
  </ports>
  <protocols>
    <protocol>http</protocol>
  </protocols>
  <notes>Any additional context the user provided</notes>
  <confidence>0.0-1.0</confidence>
</service>

EXAMPLES:

User: "Oh that's my NAS, it runs Plex and file sharing"
<service>
  <name>NAS</name>
  <ip>192.168.168.215</ip>
  <ports>
    <port>32400</port>
    <port>445</port>
  </ports>
  <protocols>
    <protocol>http</protocol>
    <protocol>smb</protocol>
  </protocols>
  <notes>Runs Plex and file sharing</notes>
  <confidence>0.95</confidence>
</service>

User: "Yeah that's the printer"
<service>
  <name>Printer</name>
  <ip>192.168.168.215</ip>
  <ports></ports>
  <protocols></protocols>
  <notes></notes>
  <confidence>0.9</confidence>
</service>

User: "I don't know what that is"
<service>
  <name></name>
  <ip>192.168.168.215</ip>
  <ports></ports>
  <protocols></protocols>
  <notes>User doesn't recognize this device - may need investigation</notes>
  <confidence>0.0</confidence>
</service>
"""

CONFIG_CHANGE_INTENT_PROMPT = """Determine if the user wants to change a security configuration setting.

USER'S MESSAGE:
"{user_message}"

CONVERSATION CONTEXT:
{context}

CONFIGURABLE SETTINGS:
- quiet_hours: When not to announce non-critical alerts (e.g., "22:00 to 08:00")
- min_severity: Minimum severity for voice alerts (critical, high, medium, low)
- check_interval: How often to check for security issues (in minutes)
- known_services: Add/remove known network devices

Output XML:
<config_intent>
  <is_config_change>true|false</is_config_change>
  <setting_category>quiet_hours|min_severity|check_interval|known_services or empty</setting_category>
  <intent>set|add|remove|query or empty</intent>
  <extracted_value>the value to set, or empty</extracted_value>
  <needs_clarification>true|false</needs_clarification>
  <clarification_question>Question to ask if unclear, or empty</clarification_question>
</config_intent>

EXAMPLES:

User: "Don't bother me with security stuff between 10pm and 7am"
<config_intent>
  <is_config_change>true</is_config_change>
  <setting_category>quiet_hours</setting_category>
  <intent>set</intent>
  <extracted_value>
    <start>22:00</start>
    <end>07:00</end>
  </extracted_value>
  <needs_clarification>false</needs_clarification>
  <clarification_question></clarification_question>
</config_intent>

User: "Only tell me about critical security issues"
<config_intent>
  <is_config_change>true</is_config_change>
  <setting_category>min_severity</setting_category>
  <intent>set</intent>
  <extracted_value>critical</extracted_value>
  <needs_clarification>false</needs_clarification>
  <clarification_question></clarification_question>
</config_intent>

User: "Add my Ring doorbell to known devices, it's at 192.168.168.220"
<config_intent>
  <is_config_change>true</is_config_change>
  <setting_category>known_services</setting_category>
  <intent>add</intent>
  <extracted_value>
    <name>Ring Doorbell</name>
    <ip>192.168.168.220</ip>
  </extracted_value>
  <needs_clarification>false</needs_clarification>
  <clarification_question></clarification_question>
</config_intent>

User: "Check for security issues more often"
<config_intent>
  <is_config_change>true</is_config_change>
  <setting_category>check_interval</setting_category>
  <intent>set</intent>
  <extracted_value></extracted_value>
  <needs_clarification>true</needs_clarification>
  <clarification_question>How often would you like me to check? Every 5 minutes, 10 minutes, or something else?</clarification_question>
</config_intent>

User: "What security alerts have there been?"
<config_intent>
  <is_config_change>false</is_config_change>
  <setting_category></setting_category>
  <intent></intent>
  <extracted_value></extracted_value>
  <needs_clarification>false</needs_clarification>
  <clarification_question></clarification_question>
</config_intent>
"""
