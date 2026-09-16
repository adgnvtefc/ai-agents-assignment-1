"""Tool definitions exposed to the model, in the OpenAI tool-calling format."""

EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": (
            "Run a bash command and return its stdout, stderr, and exit code. "
            "A non-zero exit code is reported, not raised.\n"
            "\n"
            "Every command runs in a new subshell, so a `cd` or an export does not "
            "carry over to the next command. Use the `cwd` and `env` arguments "
            "instead. Files you write do persist.\n"
            "\n"
            "Commands are non-interactive and cannot prompt for input, so pass "
            "flags like `-y` where a command would otherwise ask for confirmation. "
            "Prefer commands that produce little output; when reading a file, use "
            "`head`, `tail`, or `sed -n '10,20p'` rather than printing all of it.\n"
            "\n"
            "Useful patterns:\n"
            "- Create a file: `cat <<'EOF' > newfile.py` ... `EOF`\n"
            "- Edit in place: `sed -i 's/old/new/g' filename.py` (drop the trailing "
            "`g` to replace only the first match; restrict to a line range with "
            "`sed -i '1,10s/old/new/g'`)\n"
            "- View numbered lines: `nl -ba filename.py | sed -n '10,20p'`"
        ),
        # The nested env object intentionally accepts arbitrary variable names,
        # which is incompatible with strict schemas on some providers.
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "anyOf": [
                        {
                            "type": "string",
                            "description": 'A shell command line, e.g. "ls -la | head".',
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                'The command as an argv list, e.g. ["ls", "-la"]. '
                                "Use this with shell=false when arguments contain "
                                "characters the shell would interpret."
                            ),
                        },
                    ],
                    "description": "The command to run.",
                },
                "shell": {
                    "type": ["boolean", "null"],
                    "description": (
                        "Whether to run the command through a shell, which enables "
                        "pipes, redirection, and globbing. Defaults to true. Set to "
                        "false when passing an argv list."
                    ),
                },
                "cwd": {
                    "type": ["string", "null"],
                    "description": (
                        "Absolute path to run the command in. Defaults to the "
                        "sandbox's current working directory."
                    ),
                },
                "timeout": {
                    "type": ["number", "null"],
                    "description": (
                        "Seconds to allow the command to run before killing it. "
                        "Defaults to no timeout."
                    ),
                },
                "env": {
                    "type": ["object", "null"],
                    "additionalProperties": {"type": "string"},
                    "description": "Extra environment variables to set for this command.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

SEND_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": ("Send a message to the user."),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": ("Content of the message"),
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
}

INVOKE_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "invoke_skill",
        "description": (
            "Load a skill and return its instructions. A skill is a short guide "
            "for one kind of work, written ahead of time.\n"
            "\n"
            "Call this before starting work a skill covers, and follow what it "
            "says in place of your default approach."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "The skill's directory name, for example `hello-skill`."
                    ),
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

# TODO(3.1.a): Define an OpenAI function-tool schema named ``play_move``.
# It must accept exactly one required string argument named ``move``, explain
# that moves use UCI notation (for example e2e4), and reject extra arguments.
PLAY_MOVE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "play_move",
        "description": (
            "Play one move as White on the live board and return the position "
            "that results, with the opponent's reply already made.\n"
            "\n"
            "This changes the real game. Call it once per turn, with the move "
            "you have decided on. An illegal move, a move out of turn, or a "
            "move in a finished game is reported back to you and costs the "
            "turn nothing, but the board is otherwise committed.\n"
            "\n"
            "If you chose the move by searching inside `run_python`, commit it "
            "there instead: `play_move(best)` as the snippet's last statement "
            "is this same tool, applied to the same board. Printing the move "
            "and then calling this tool separately re-decides it without the "
            "search in front of you, and is not how a move should be played."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": (
                        "The move in UCI notation: the square moved from "
                        "followed by the square moved to, for example `e2e4` "
                        "or `g1f3`. Promotions add the piece letter, as in "
                        "`e7e8q`."
                    ),
                },
            },
            "required": ["move"],
            "additionalProperties": False,
        },
    },
}

# TODO(3.3): Define the `simulate_move` tool, like the `play_move` tool.
SIMULATE_MOVE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "simulate_move",
        "description": (
            "Search without committing. Returns a position and its legal "
            "moves, leaving the real game untouched.\n"
            "\n"
            "With a FEN alone, describes that position. With a FEN and a move, "
            "plays exactly that one ply — for either colour — and describes "
            "the position after it. No opponent reply is made.\n"
            "\n"
            "Chain it to look ahead: feed the `fen` it returns back in to go "
            "another ply deeper, as many times as you like. Use this for every "
            "candidate you are weighing, then `play_move` once for the move "
            "you chose."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "fen": {
                    "type": "string",
                    "description": (
                        "The position to inspect, as a complete six-field FEN "
                        "(placement, side to move, castling, en passant, "
                        "halfmove clock, fullmove number). Take it from the "
                        "`fen` of an observation or of an earlier simulation."
                    ),
                },
                "move": {
                    # Strict mode requires every property in `required`, so an
                    # optional argument is expressed as a nullable one: the
                    # model passes null to inspect the position as it stands.
                    "type": ["string", "null"],
                    "description": (
                        "Optional. One move in UCI notation, for example "
                        "`e2e4` or `e7e8q`, to play from that position. Pass "
                        "null to inspect the position without moving."
                    ),
                },
            },
            "required": ["fen", "move"],
            "additionalProperties": False,
        },
    },
}

# TODO()
RUN_PYTHON_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run a Python snippet next to the chess server and return what it "
            "printed.\n"
            "\n"
            "`simulate_move(fen, move=None)` and `play_move(move)` are already "
            "defined in the snippet's namespace as ordinary synchronous "
            "functions returning dicts. Do not import them, do not define "
            "them, and do not wrap them in tool-call syntax -- just call "
            "them. A rejected call raises instead of returning, so a bad move "
            "surfaces at the line that made it.\n"
            "\n"
            "Use this to search properly: loop over candidate moves, score "
            "them with `simulate_move`, and pick the best. Only what you "
            "`print()` comes back, so print the result you need to see.\n"
            "\n"
            "End the snippet by playing what you chose: `play_move(best)` as "
            "the last statement. That call IS the `play_move` tool -- same "
            "board, same commit, same returned position -- so a snippet that "
            "ends there has played the move for this turn and needs no "
            "separate tool call. A snippet that only prints its answer has "
            "changed nothing and wasted the search.\n"
            "\n"
            "The snippet is stopped if it runs too long, so keep the search "
            "bounded: a fixed list of candidates, a fixed depth, no unbounded "
            "loops."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "The Python source to execute. Runs top to bottom in "
                        "one namespace; the standard library is available."
                    ),
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}
