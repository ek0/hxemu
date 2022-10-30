# hxemu: Symbolic emulation through Triton

`hxemu` is a small library that leverages triton to perform symbolic execution in an arbitrary context.
As opposed to other emulators, like Unicorn or bochscpu, triton allows us to emulate code with partial contexts,
or no context information at all. Unknown registers or memory cells are considered symbolic and the emulator state
can be altered at will for analysis.

## Features

- Ability to run in unknown contexts.
- User defined interface to run traces in said arbitrary context.

## Building

We currently only support Windows, although porting it to other platforms should be trivial with minimal modifications, if any. You can build `hxemu` using cmake as follow:

```
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

## Usage

Here is a quick example:

```cpp
#include <hxemu.h>

class MyHook : public EmulatorHookInterface
{
public:
	void OnEmulateExit(Emulator& emulator, triton::arch::Instruction& instruction) override
	{
		// Do something after the instruction is emulated...
	}

	// ... Other callbacks
};

// Class building instruction resolved from the process address space
// This logic can be applied for small, incomplete traces, debuggee and even IDBs
class InstructionFactory : public InstructionFactoryInterface
{
public:
	std::optional<triton::arch::Instruction> FromAddress(uint64_t address)
	{
		// Build instruction object...
	}
}

bool Emulate(ea_t start_address)
{
	std::unique_ptr<InstructionFactory> factory = std::make_unique<InstructionFactory>();
	Emulator emulator(triton::arch::ARCH_X86_64, std::move(factory));

	std::shared_ptr<MyHook> hooks = std::make_shared<MyHook>();
	return emulator.Run(start_address, hooks);
}
```

The main emulation entry points are:
- `EmulateOnInstruction` which emulates only one instruction in the current emulator context.
- `Run` which emulates until RIP is unknown (meaning it doesn't have a concrete value).

## Known Issues

Since `triton` symbolic engine is quite slow and heavy, emulation will be slow. It can also take a lot of your memory, even for small traces.