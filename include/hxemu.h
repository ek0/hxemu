#pragma once

#include <triton/context.hpp>
#include <llvm/IR/Module.h>

#include <pro.h>

#include <optional>

namespace hxemu
{

class Emulator; // Required to declare EmulatorHookInterface

// Class used to execute code after specific event during the emulation.
// To use this, please inherit from this class and see `EmulateUntilSymbolic`
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

class Emulator
{
public:
    // FIXME: Implement support for multiple platform later on.
    Emulator() : ctx_(triton::arch::architecture_e::ARCH_X86_64) {}

    // Initialize the emulator. Emulation will start at `start_address` and RIP will be concrete.
    void Initialize(ea_t start_address);

    // Symbolize all registers
    void SymbolizeAllRegister();

    // Resets the emulator context
    void ResetContext();

    // Emulate one instruction
    bool EmulateOneInstruction(triton::arch::Instruction& address);

    // Emulate code until PC is symbolic
    bool EmulateUntilSymbolic(ea_t start_address, std::shared_ptr<EmulatorHookInterface> hooks = nullptr, size_t max_instructions_to_process = 300000);

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

    // Get the memory value at the address stored in `reg`
    //std::optional<triton::uint512> GetMemoryValue(triton::arch::register_e reg);

    // Get the memory value for the address stored in `reg`, if any
    std::optional<triton::uint512> GetMemoryValue(triton::arch::register_e reg, size_t size);

    // Retrieve the symbolic memory expression for memory address stored in `reg`
    const triton::engines::symbolic::SharedSymbolicExpression GetSymbolicMemoryExpression(triton::arch::register_e reg);

    // Symbolize the specified register, thus converting the current expression into a symbolic variable
    triton::engines::symbolic::SharedSymbolicVariable SymbolizeRegister(const triton::arch::Register& reg, const std::string& alias = "");

    // Convert AST to LLVM bitcode
    std::shared_ptr<llvm::Module> ConvertToLLVM(const triton::ast::SharedAbstractNode node);

    // Is the specified register symbolic?
    bool IsSymbolic(const triton::arch::Register& reg) const;

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
    std::optional<triton::arch::Instruction> FromAddress(ea_t address);

    // Emulation context
    triton::Context ctx_;
};

} // namespace hxemu