from aiolaunchpad import LaunchControlXL

pad = LaunchControlXL()

@pad.register
async def print_fader_0(code, input: pad.inputs.fader0, value, device):
    print("Hello, fader 0 has value", value)

@pad.register
async def print_all_faders(code, input, value, device):
    print("Any fader prints this")

if __name__ == '__main__':
    pad.run_app()