#pragma once

#include <triton/context.hpp>
#include <llvm/IR/Module.h>

#include <optional>

namespace hxemu
{

class Emulator; // Required to declare EmulatorHookInterface

// Class used to execute code after specific event during the emulation.
// To use hooks, the user must inherit from this class and implement its methods.
// See `Run` for more information.
class EmulatorHookInterface
{
public:
    virtual ~EmulatorHookInterface() {};

    // Callback executing after an instruction is executed.
    virtual void OnEmulateExit(Emulator& emulator, triton::arch::Instruction& instruction) = 0;

    // Callback executing before an instruction is executed.
    // Symbolic context hasn't been updated. Instruction disassembly has been populated.
    virtual void OnEmulateEnter(Emulator& emulator, triton::arch::Instruction& instruction) = 0;
};

// Interface to build triton instruction in the context of the running application.
// The user must inherit from the class and implement the CreateInstruction method.
// This can rely on any source: actual memory, partial serialized traces, an IDB...
class InstructionFactoryInterface
{
public:
    virtual ~InstructionFactoryInterface() {};

    // Create a triton instruction object for the given address
    virtual std::optional<triton::arch::Instruction> FromAddress(uint64_t address) = 0;
};

// Adapter around the triton::Context object.
// It provides an simpler interface to query the context through optionals.
// It initializes the Context object in a proper way to run with best performances.
class Emulator
{
public:
    // FIXME: Implement support for multiple platform later on.
    Emulator(triton::arch::architecture_e arch, std::unique_ptr<InstructionFactoryInterface> instruction_factory)
        : ctx_(arch), initialized_(false), factory_(nullptr)
    {
        if (instruction_factory == nullptr)
            throw std::exception("Invalid InstructionBuilder pointer used during emulator initialization");

        factory_ = std::move(instruction_factory);
        // Start with symbolizing all registers...
        //SymbolizeAllRegister(); // TODO: Let the user define this.
        // Set triton mode. These ones are important for performance purposes.
        ctx_.setMode(triton::modes::AST_OPTIMIZATIONS, true);
        ctx_.setMode(triton::modes::CONSTANT_FOLDING, true);
        ctx_.setMode(triton::modes::MEMORY_ARRAY, true);
        switch (arch)
        {
        case triton::arch::ARCH_X86_64:
            pc_id_ = triton::arch::ID_REG_X86_RIP;
            break;
        case triton::arch::ARCH_X86:
            pc_id_ = triton::arch::ID_REG_X86_EIP;
            break;
        case triton::arch::ARCH_ARM32:
            pc_id_ = triton::arch::ID_REG_ARM32_PC;
            break;
        case triton::arch::ARCH_AARCH64:
            pc_id_ = triton::arch::ID_REG_AARCH64_PC;
            break;
        default:
            throw std::exception("Unsupported architecture for emulator");
        }
        // Prevent automatic concretization of memory loads. We don't have to use a hack anymore (?)
        //ctx_.setMode(triton::modes::SYMBOLIZE_LOAD, true);
        initialized_ = true;
    }

    // Initialize the emulator. Emulation will start at `start_address` and RIP will be concrete.
    //void Initialize();

    // Set a callback for every memory load encountered.
    void SetOnLoadMemoryCallback(triton::callbacks::getConcreteMemoryValueCallback callback);

    // Set a callback for every memory store encountered.
    void SetOnStoreMemoryCallback(triton::callbacks::setConcreteMemoryValueCallback callback);

    // Set the triton::Context mode.
    void SetMode(triton::modes::mode_e mode);

    // Set the representation mode for output
    void SetRepresentationMode(triton::ast::representations::mode_e mode);

    // Symbolize all registers
    void SymbolizeAllRegister();

    // Resets the emulator context
    void ResetContext();

    // Emulate one instruction
    bool EmulateOneInstruction(triton::arch::Instruction& address);

    // Emulate code until PC is symbolic
    bool Run(uint64_t start_address, std::shared_ptr<EmulatorHookInterface> hooks = nullptr, size_t max_instructions_to_process = 300000);

    // Slice expressions
    std::unordered_map<uint64_t, triton::engines::symbolic::SharedSymbolicExpression> SliceExpression(const triton::engines::symbolic::SharedSymbolicExpression& expr);

    // Transform a symbolic expression into a new symbolic variable.
    void SymbolizeExpression(const triton::engines::symbolic::SharedSymbolicExpression& expr);

    // Returns the register object
    // TODO: This kindof leak the internals of the emulator, albeit const.
    //       This is due to the way triton passes things around...
    const triton::arch::Register& GetRegister(triton::arch::register_e reg) const;

    // Retrieve the concrete value for the given register id
    std::optional<triton::uint512> GetRegisterValue(triton::arch::register_e reg);

    // Retrieve the concrete value for the given register object
    std::optional<triton::uint512> GetRegisterValue(const triton::arch::Register& reg);

    // Retrieve the register AST from the current context
    triton::ast::SharedAbstractNode GetRegisterAst(triton::arch::register_e reg);

    // Retrieve the register AST from the register object
    triton::ast::SharedAbstractNode GetRegisterAst(const triton::arch::Register& reg);

    // Retrieve the memory AST
    triton::ast::SharedAbstractNode GetMemoryAst(const triton::arch::MemoryAccess& mem);

    // Retrieve the specified operand AST
    triton::ast::SharedAbstractNode GetOperandAst(const triton::arch::OperandWrapper& op);

    // Retrieve the register AST from the register object
    //triton::ast::SharedAbstractNode GetRegisterAst(const triton::arch::Register& reg) const;

    // Set the given register ID to the specified value.
    void SetRegisterValue(triton::arch::register_e reg, uint64_t value);

    // Retrieve the symbolic register expression
    const triton::engines::symbolic::SharedSymbolicExpression& GetSymbolicRegisterExpression(triton::arch::register_e reg);

    // Retrieve the symbolic register expression
    const triton::engines::symbolic::SharedSymbolicExpression& GetSymbolicRegisterExpression(const triton::arch::Register& reg);

    // Retrieve the concrete value for the given memory location
    std::optional<std::vector<uint8_t>> GetMemoryValue(uint64_t address, size_t size);

    // Retrieve the concrete value for the given memory location
    std::optional<triton::uint512> GetMemoryValue(triton::arch::MemoryAccess& mem);

    // Get the memory value for the address stored in `reg`, if any
    std::optional<triton::uint512> GetMemoryValue(triton::arch::register_e reg, size_t size);

    // Retrieve the symbolic memory expression for memory address stored in `reg`
    const triton::engines::symbolic::SharedSymbolicExpression GetSymbolicMemoryExpression(triton::arch::register_e reg);

    // Symbolize the specified register, thus converting the current expression into a symbolic variable
    triton::engines::symbolic::SharedSymbolicVariable SymbolizeRegister(const triton::arch::Register& reg, const std::string& alias = "");

    // Symbolize the specified register, thus converting the current expression into a symbolic variable
    triton::engines::symbolic::SharedSymbolicVariable SymbolizeRegister(const triton::arch::register_e reg, const std::string& alias = "");

    // Convert AST to LLVM bitcode
    std::shared_ptr<llvm::Module> ConvertToLLVM(const triton::ast::SharedAbstractNode node);

    // Is the specified register symbolic?
    bool IsSymbolic(const triton::arch::Register& reg) const;

    // Is the specified register symbolic
    bool IsSymbolic(const triton::arch::register_e reg) const;

    // For smaller types
    // Returns the memory value for the address stored in `reg`, if any.
    template<typename T>
    std::optional<T> GetMemoryValue(triton::arch::register_e reg)
    {
        static_assert(std::is_integral_v<T>, "Value is not integral");
        const triton::arch::Register& r = ctx_.getRegister(reg);
        const triton::ast::SharedAbstractNode register_ast = ctx_.getRegisterAst(r);
        if (register_ast->isSymbolized())
            return std::nullopt; // One symbolic variable is found in the expression. Can't find the address
        uint64_t address = static_cast<uint64_t>(register_ast->evaluate());
        if (!ctx_.isConcreteMemoryValueDefined(address, sizeof(T)) || ctx_.isMemorySymbolized(address, sizeof(T)))
            return std::nullopt;
        return static_cast<T>(ctx_.getConcreteMemoryValue(triton::arch::MemoryAccess(address, sizeof(T))));
    };

private:
    // Emulation context
    triton::Context ctx_;

    std::function<triton::arch::Instruction(uint64_t)> FromAddress;

    // True if the emulator has properly initialized
    bool initialized_;

    // Actual identifier for the PC register. This can change if emulating different architectures.
    triton::arch::register_e pc_id_;

    std::unique_ptr<InstructionFactoryInterface> factory_;

    // Lifetime of the context should be the same as the targetted emulation context
    llvm::LLVMContext llvm_context_;
};

} // namespace hxemu