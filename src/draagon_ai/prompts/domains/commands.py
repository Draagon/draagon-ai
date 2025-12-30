"""Commands domain prompts.

Shell command generation and error recovery prompts.
"""

COMMAND_GENERATION_PROMPT = """Generate a shell command to answer this question. Be OS-aware.

QUESTION: {question}
TARGET HOST: {host}

SYSTEM INFO:
{system_info}

COMMAND GUIDELINES:
- PREFER SIMPLE SINGLE COMMANDS - avoid chaining with && or || unless necessary
- For package queries on Debian/Ubuntu: use `dpkg -l | grep -i [name]`
- For service status: use `systemctl is-active [service]` (simplest) or `systemctl status [service]`
- For checking if docker is running: use `systemctl is-active docker`
- For disk space: use `df -h`
- For memory: use `free -h`
- For processes: use `pgrep -a [name]` or `ps aux | grep -i [name]`
- For checking if installed: use `which [program]`
- Always limit output with `| head -N` for long listings
- Do NOT use docker --format with Go template syntax ({{{{.Name}}}}) - just use plain docker ps

SAFETY: Only generate read-only commands. Never generate destructive commands.

Output XML:
<command_generation>
    <command>the shell command</command>
    <explanation>what this will show</explanation>
</command_generation>

Examples:
Question: "What VNC software is installed?"
System: Pop!_OS (Ubuntu-based)
-> <command_generation><command>dpkg -l | grep -i vnc</command><explanation>Lists all installed packages containing 'vnc'</explanation></command_generation>

Question: "Is docker running?"
-> <command_generation><command>systemctl is-active docker</command><explanation>Checks if docker service is active</explanation></command_generation>

Question: "What docker containers are running?"
-> <command_generation><command>docker ps</command><explanation>Lists running docker containers</explanation></command_generation>"""


ERROR_RECOVERY_PROMPT = """A command failed. Analyze the error and suggest a fix or alternative.

ORIGINAL QUESTION: {question}
ATTEMPTED COMMAND: {command}
ERROR OUTPUT: {error}
HOST: {host}

SYSTEM INFO:
{system_info}

ANALYSIS GUIDELINES:
1. "command not found" - suggest installing the package or an alternative command
2. "permission denied" - suggest if sudo is needed (but warn about security)
3. "no such file or directory" - suggest correct path or alternative
4. Empty output - might mean "not installed" or "not found" which IS an answer
5. Timeout - suggest simpler command

Output XML:
<error_recovery>
    <can_retry>true or false</can_retry>
    <new_command>alternative command if can_retry, or empty</new_command>
    <answer>answer if we can determine it from error, or empty</answer>
    <explanation>what went wrong</explanation>
</error_recovery>

Examples:
Error: "grep: command not found"
-> <error_recovery><can_retry>true</can_retry><new_command>dpkg -l | head -50</new_command><explanation>grep not available, showing all packages instead</explanation></error_recovery>

Error: "" (empty output for "dpkg -l | grep vnc")
-> <error_recovery><can_retry>false</can_retry><answer>No VNC packages are currently installed on this system.</answer><explanation>Empty output means no matching packages</explanation></error_recovery>"""


COMMANDS_PROMPTS = {
    "COMMAND_GENERATION_PROMPT": COMMAND_GENERATION_PROMPT,
    "ERROR_RECOVERY_PROMPT": ERROR_RECOVERY_PROMPT,
}
