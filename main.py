from cli_core import CLI, discover_commands

discover_commands('text', 'gen_image', 'search', 'gen_video')  # import side-effect: registers cmds

cli = CLI(
    prog='ai-tools',
    description='API-based Swiss-army knife for AI',
    epilog='Run "ai-tools <command> --help" for command-specific flags',
)
cli.run()