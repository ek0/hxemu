# hxemu: Symbolic emulation through Triton in IDA Pro

`hxemu` is a small library that leverages triton to perform symbolic execution in the current IDB context.
As opposed to other emulators, like Unicorn and bochscpu, triton allows us to emulate code in unknown contexts
and preserve this information.

## Features

- Ability to run in unknown contexts.
- Atuomatically resolve concrete memory if mapped in the current IDB.

## Building

We currently only support Windows, although porting it to other platforms should be trivial. You can build `hxemu` using cmake as follow:

```
cmake -G "Ninja" -DIDA_SDK=<path to the IDA SDK> ..
cmake --build .
```

## Usage

Here is a quick example:

```cpp
#include <hxemu.h>

class MyHook : public EmulatorHookInterface
{
public:
	void OnEmulateInstructionAfter(Emulator& emulator, triton::arch::Instruction& instruction) override
	{
		// Do something after the instruction is emulated...
	}
};

bool Emulate(ea_t start_address)
{
	Emulator emulator;
	emulator.Initialize(start_address); // Initialize the symbolic context and the callbacks.

	std::unique_ptr<MyHook> hooks = std::make_unique<MyHook>();
	return emulator.EmulateUntilSymbolic(start_address, std::move(hooks));
}
```

The main emulation entry points are:
- `EmulateOnInstruction` which emulates only one instruction in the current emulator context.
- `EmulateUntilSymbolic` which emulates until RIP is unknown (meaning it doesn't have a concrete value).

## Known Issues

Since `triton` symbolic engine is quite slow and heavy, emulation will be slow. It can also take a lot of your memory, even for small traces.