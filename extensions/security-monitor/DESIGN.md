# Security Monitor Extension Design

**Extension:** `draagon-ai-ext-security`
**Version:** 0.1.0
**Author:** Doug
**Created:** 2025-12-29

---

## Overview

A background security monitoring extension for draagon-ai that provides autonomous, AI-powered security analysis with agentic investigation capabilities.

### Key Features

1. **Multi-Source Monitoring** - Suricata IDS, syslogs, network health
2. **Groq 70B Analysis** - Fast, intelligent threat classification
3. **Agentic Investigation** - ReAct-style deep investigation when needed
4. **Memory Integration** - Remembers past issues, learns from false positives
5. **Voice Announcements** - Speaks alerts through Home Assistant voice devices
6. **Extensible Architecture** - Easy to add new monitors

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SecurityMonitorExtension                             │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Monitor Registry                             │   │
│  │                                                                  │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │   │
│  │  │  Suricata    │ │   Syslog     │ │   System     │   ...more   │   │
│  │  │  Monitor     │ │   Monitor    │ │   Health     │             │   │
│  │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘             │   │
│  │         │                │                │                      │   │
│  │         └────────────────┴────────────────┘                      │   │
│  │                          │                                       │   │
│  └──────────────────────────┼───────────────────────────────────────┘   │
│                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Alert Aggregator                              │   │
│  │                                                                  │   │
│  │  • Deduplication (same alert within window)                      │   │
│  │  • Enrichment (GeoIP, DNS reverse lookup)                        │   │
│  │  • Batching (group related alerts)                               │   │
│  │  • Priority queue (critical first)                               │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  Security Analyzer (Agentic)                     │   │
│  │                                                                  │   │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐          │   │
│  │  │  PLAN   │ → │  THINK  │ → │   ACT   │ → │ REFLECT │          │   │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘          │   │
│  │       │                           │                              │   │
│  │       │         Groq 70B          │      Security Tools          │   │
│  │       │    (llama-3.3-70b)        │    (threat intel, memory)    │   │
│  │       │                           │                              │   │
│  └───────┼───────────────────────────┼──────────────────────────────┘   │
│          │                           │                                  │
│          ▼                           ▼                                  │
│  ┌───────────────┐           ┌───────────────┐                         │
│  │    Memory     │           │   Notifier    │                         │
│  │   (Qdrant)    │           │  (HA Voice)   │                         │
│  └───────────────┘           └───────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Monitors

Abstract base class with concrete implementations for each data source.

```python
class BaseMonitor(ABC):
    """Base class for all security monitors."""

    @abstractmethod
    async def check(self) -> list[Alert]:
        """Check for new alerts since last check."""
        pass

    @abstractmethod
    def get_status(self) -> MonitorStatus:
        """Get current monitor health status."""
        pass
```

#### Suricata Monitor
- Watches `/var/log/suricata/eve.json`
- Filters for `event_type: alert`
- Tracks last read position to avoid duplicates
- Parses signature, severity, src/dest IPs, protocol

#### Syslog Monitor
- Watches `/var/log/auth.log` for auth failures
- Watches `/var/log/syslog` for critical errors
- Pattern matching for security-relevant events

#### System Health Monitor
- NVMe temperature (SMART data)
- Disk usage thresholds
- Memory pressure
- Service health (systemd units)

### 2. Alert Aggregator

Processes raw alerts before analysis:

```python
@dataclass
class Alert:
    id: str
    source: str                    # "suricata", "syslog", "system"
    timestamp: datetime
    severity: Severity             # critical, high, medium, low, info
    signature: str                 # Alert signature/title
    description: str               # Human-readable description
    source_ip: str | None
    dest_ip: str | None
    protocol: str | None
    raw_data: dict                 # Original event data
    enrichments: dict              # GeoIP, DNS, etc.
```

**Aggregation Rules:**
- Same signature + same source IP within 5 minutes → dedupe
- Group related alerts (same source IP, different signatures)
- Batch processing every 60 seconds or 10 alerts

### 3. Security Analyzer

The brain of the extension - uses Groq 70B with agentic reasoning.

**Analysis Flow:**

1. **Memory Check** - Have we seen this before?
   ```python
   past_issues = await memory.search(f"security alert {signature}", limit=5)
   if is_known_false_positive(alert, past_issues):
       return AnalysisResult(threat_level="noise", action="suppress")
   ```

2. **Quick Classification** - Fast Groq call for obvious cases
   ```python
   result = await groq.classify(alert, context=network_context)
   if result.confidence > 0.9:
       return result
   ```

3. **Agentic Investigation** - For uncertain cases
   ```python
   trace = await agentic_loop.process(
       query=f"Investigate: {alert.signature} from {alert.source_ip}",
       tools=security_tools,
   )
   ```

### 4. Security Tools

Tools available to the agentic analyzer:

| Tool | Description |
|------|-------------|
| `check_suricata_alerts` | Get recent alerts by severity/time |
| `search_threat_intel` | Query SearXNG for threat info |
| `check_ip_reputation` | Lookup IP in AbuseIPDB/similar |
| `query_pihole_logs` | Check DNS queries for a domain |
| `recall_security_memory` | Search past investigations |
| `check_firewall_rules` | Verify blocking rules |
| `get_network_context` | Current network state |

### 5. Notifier

Sends alerts through appropriate channels based on severity.

```python
class NotificationChannel(ABC):
    @abstractmethod
    async def send(self, message: str, priority: Priority) -> bool:
        pass

class HAVoiceChannel(NotificationChannel):
    """Announce through Home Assistant Voice PE device."""

    async def send(self, message: str, priority: Priority) -> bool:
        await ha.call_service(
            domain="tts",
            service="speak",
            data={
                "entity_id": self.voice_entity,
                "message": message,
                "cache": False,
            }
        )
```

**Notification Rules:**

| Threat Level | Voice | Log | Memory |
|--------------|-------|-----|--------|
| Critical | Immediate + repeat | Yes | Yes |
| High | Immediate | Yes | Yes |
| Medium | Queue for summary | Yes | Yes |
| Low | Silent | Yes | Optional |
| Noise | Silent | Optional | Learn |

---

## Configuration

```yaml
# draagon.yaml or extension config
security-monitor:
  # Check interval
  check_interval_seconds: 300

  # LLM Configuration
  llm:
    provider: groq
    model: llama-3.3-70b-versatile
    api_key: ${GROQ_API_KEY}
    temperature: 0.1
    max_tokens: 1000

  # Network Context (for accurate analysis)
  network:
    home_net: "192.168.168.0/24"
    known_services:
      - name: "Plex"
        ip: "192.168.168.204"
      - name: "Home Assistant"
        ip: "192.168.168.206"
      - name: "Pi-hole"
        ip: "192.168.168.208"
      - name: "NAS"
        ip: "192.168.168.202"

  # Monitors
  monitors:
    suricata:
      enabled: true
      eve_log: /var/log/suricata/eve.json
      min_severity: low

    syslog:
      enabled: true
      paths:
        - /var/log/auth.log
        - /var/log/syslog
      patterns:
        - "Failed password"
        - "authentication failure"
        - "CRITICAL"

    system_health:
      enabled: true
      checks:
        nvme_temp:
          enabled: true
          warning_celsius: 60
          critical_celsius: 70
        disk_usage:
          enabled: true
          warning_percent: 85
          critical_percent: 95

  # Notifications
  notifications:
    voice:
      enabled: true
      entity_id: "assist_satellite.home_assistant_voice_0a5786_assist_satellite"
      min_severity: high
      quiet_hours:
        start: "22:00"
        end: "08:00"

    summary:
      enabled: true
      schedule: "09:00"  # Daily summary
      include_noise: false

  # Memory
  memory:
    collection: "security_investigations"
    learn_from_noise: true
    retention_days: 90
```

---

## Prompts

### Classification Prompt

```
You are a security analyst for a HOME network (not enterprise).

NETWORK CONTEXT:
{network_context}

ALERT:
{alert_json}

PAST SIMILAR ISSUES:
{memory_context}

Common FALSE POSITIVES in home networks:
- ET POLICY: Software updates (apt, snap, brew)
- ET INFO: Streaming services, CDNs
- ET SCAN: Shodan, Censys, academic scanners
- ET DNS: Ad networks, analytics (handled by Pi-hole)

Classify this alert:
{
    "threat_level": "critical|high|medium|low|noise",
    "is_false_positive": boolean,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "action_required": "What to do (if anything)",
    "needs_investigation": boolean
}
```

### Investigation Prompt

```
You are investigating a security alert on Doug's home network.

ALERT DETAILS:
{alert}

YOUR GOAL:
Determine if this is a real threat or false positive.

AVAILABLE TOOLS:
{tools}

INVESTIGATION STEPS:
1. Check if we've seen this IP/signature before
2. Look up threat intelligence if unknown
3. Check if our firewall/Pi-hole is already handling it
4. Determine if any action is needed

Be thorough but efficient. Most home network alerts are noise.
```

### Voice Announcement Prompt

```
Convert this security finding into a brief, natural voice announcement.

FINDING:
{finding_json}

RULES:
- Keep under 30 seconds of speech
- Lead with severity ("Security alert" vs "Quick update")
- Explain in plain English (no IP addresses unless critical)
- End with action required or "no action needed"
- Be conversational, not robotic

EXAMPLES:
- Critical: "Security alert. I detected multiple failed login attempts targeting your server. I've verified your firewall is blocking them, but you should check your fail2ban logs."
- Low: "Quick security update. I noticed some routine port scanning from a known research scanner. Your firewall blocked it. No action needed."
```

---

## Integration with Roxy

### New Tool: `security_status`

```python
Tool(
    name="security_status",
    description="Get current security status and recent alerts",
    parameters=[
        Parameter("timeframe", "string", "How far back to look",
                  enum=["1h", "24h", "7d"], default="24h"),
        Parameter("include_noise", "boolean", "Include low-priority alerts",
                  default=False),
    ],
)
```

### Roxy Queries

```
User: "Roxy, any security issues?"
→ Calls security_status tool
→ Returns natural language summary

User: "Tell me about that SSH scan"
→ Recalls from memory
→ Provides detailed explanation

User: "Investigate the alerts from yesterday"
→ Triggers agentic investigation loop
→ Returns comprehensive report
```

---

## Background Service

The monitor runs as a background task within the extension:

```python
class SecurityMonitorService:
    """Background service that continuously monitors for threats."""

    async def start(self):
        """Start the monitoring loop."""
        self._running = True
        while self._running:
            try:
                # Collect alerts from all monitors
                alerts = await self._collect_alerts()

                if alerts:
                    # Aggregate and analyze
                    batches = self._aggregator.process(alerts)
                    for batch in batches:
                        result = await self._analyzer.analyze(batch)
                        await self._handle_result(result)

                # Wait for next check
                await asyncio.sleep(self._config.check_interval)

            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def stop(self):
        """Stop the monitoring loop gracefully."""
        self._running = False
```

---

## Data Flow Example

### Scenario: SSH Scan Detected

1. **Suricata logs alert** to eve.json
   ```json
   {"event_type": "alert", "alert": {"signature": "ET SCAN SSH"}, "src_ip": "45.33.32.156"}
   ```

2. **Monitor picks up alert**
   ```python
   alert = Alert(signature="ET SCAN SSH", source_ip="45.33.32.156", severity="medium")
   ```

3. **Aggregator enriches**
   ```python
   alert.enrichments = {"geo": "US/California", "reverse_dns": "scanner.shodan.io"}
   ```

4. **Analyzer checks memory**
   ```python
   past = await memory.search("45.33.32.156")
   # Found: "Known Shodan scanner, seen 5 times, always false positive"
   ```

5. **Quick classification** (no investigation needed)
   ```python
   result = AnalysisResult(
       threat_level="noise",
       is_false_positive=True,
       reasoning="Known Shodan scanner, previously marked as benign",
       action="suppress",
   )
   ```

6. **Store in memory** (reinforce learning)
   ```python
   await memory.store("45.33.32.156 - Shodan scanner - false positive confirmed")
   ```

7. **No notification** (noise level)

### Scenario: Unknown Threat

1. Same flow until step 4...

4. **Memory check returns nothing**
   ```python
   past = await memory.search("suspicious.domain.ru")
   # No results
   ```

5. **Trigger agentic investigation**
   ```python
   trace = await agentic_loop.process(
       query="Investigate connection to suspicious.domain.ru",
       tools=[search_threat_intel, check_ip_reputation, query_pihole_logs],
   )
   ```

6. **Investigation finds threat intel**
   - Web search: "Domain associated with malware C2"
   - AbuseIPDB: Confidence 95% malicious
   - Pi-hole: Blocked 47 queries to this domain

7. **Generate report**
   ```python
   result = AnalysisResult(
       threat_level="high",
       reasoning="Domain linked to malware C2, blocked by Pi-hole",
       action="Verify no malware on network devices",
   )
   ```

8. **Voice announcement**
   > "Security alert. I detected connections to a suspicious domain that's linked to malware. The good news is Pi-hole has been blocking it. You should scan your devices for malware to be safe."

9. **Store investigation in memory**

---

## Future Extensions

### Additional Monitors

```python
# Easy to add new monitors
class ProxmoxMonitor(BaseMonitor):
    """Monitor Proxmox VM/container health."""

class DockerMonitor(BaseMonitor):
    """Monitor Docker container status."""

class BackupMonitor(BaseMonitor):
    """Verify backups are running successfully."""
```

### Proactive Hunting

Instead of just reactive monitoring, periodically search for:
- Unusual outbound connections
- New devices on network
- Certificate expiration
- DNS anomalies

---

## Implementation Phases

### Phase 1: Core Framework
- [ ] Extension structure and registration
- [ ] BaseMonitor class
- [ ] SuricataMonitor implementation
- [ ] Basic Groq integration

### Phase 2: Analysis
- [ ] Alert aggregator
- [ ] Classification prompt
- [ ] Memory integration
- [ ] Basic notification (logging)

### Phase 3: Agentic Loop
- [ ] Security tools
- [ ] Investigation prompt
- [ ] ReAct loop integration
- [ ] Result synthesis

### Phase 4: Notifications
- [ ] HA Voice integration
- [ ] Voice announcement prompt
- [ ] Quiet hours
- [ ] Daily summary

### Phase 5: Roxy Integration
- [ ] security_status tool
- [ ] Natural language queries
- [ ] Cross-system memory

---

## Dependencies

```toml
[project]
dependencies = [
    "draagon-ai>=0.1.0",
    "httpx>=0.25.0",       # Async HTTP for APIs
    "aiofiles>=23.0.0",    # Async file reading
    "geoip2>=4.0.0",       # GeoIP enrichment (optional)
]

[project.optional-dependencies]
groq = ["groq>=0.4.0"]
```

---

## Testing Strategy

1. **Unit tests** for each monitor
2. **Integration tests** with mock eve.json
3. **Prompt testing** with known alert patterns
4. **End-to-end** with real Suricata data (sanitized)

```python
def test_suricata_monitor_detects_alert():
    monitor = SuricataMonitor(eve_log="tests/fixtures/eve_sample.json")
    alerts = await monitor.check()
    assert len(alerts) == 3
    assert alerts[0].signature == "ET SCAN SSH"
```
