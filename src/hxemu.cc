#include "hxemu.h"

#include <triton/tritonToLLVM.hpp>

#include <llvm/IR/LLVMContext.h>

#include <bytes.hpp>
#include <ua.hpp>

namespace hxemu
{

void OnLoadMemory(triton::Context& ctx, const triton::arch::MemoryAccess& mem)
{
    // HACK: From hxsym:
    // Triton always conrecretize load and store.
    // This can cause issues when we're processing traces that reference
    // memory in our destination register.
    // Since we rely on "Is this register symbolized?" to know if we successfully
    // computed a value or not, we needed a way to symbolize memory on demand.
    //
    // We use this callback to check if the LeaAst is symbolic (this means there's a symbolic variable
    // in the expression). And we also check if the computed memory access has a concrete value backing it.
    // If not, we'll just symbolize the resulting the memory access, which will then be propagated to other
    // expressions.
    if (ctx.isMemorySymbolized(mem))
        return;
    // Value already defined.
    if (ctx.isConcreteMemoryValueDefined(mem))
        return;
    // FIXME: Some pop/ret instruction will have a null lea ast for some reason.
    if (mem.getLeaAst() == nullptr)
    {
        ctx.symbolizeMemory(mem);
        return;
    }
    // Does the AST contain a symbolic variable? Or has the referenced memory been concretized?
    if (mem.getLeaAst()->isSymbolized())
    {
        ctx.symbolizeMemory(mem);
        return;
    }

    // Here we know the address is valid, and we can fetch the data from the IDB
    std::vector<uint8_t> memory_area(mem.getSize());
    ssize_t bytes_read = get_bytes(memory_area.data(), mem.getSize(), mem.getAddress());
    if (bytes_read > 0)
        ctx.setConcreteMemoryAreaValue(mem.getAddress(), memory_area, false); // We don't exec the callbacks here.
    else
        ctx.symbolizeMemory(mem); // Address is probably not in the IDB, maybe an external module or a wrong dump.
}

void OnStoreMemory(triton::Context& ctx, const triton::arch::MemoryAccess& mem)
{
}

void Emulator::Initialize(ea_t start_address)
{
    // Start with symbolizing all registers...
    SymbolizeAllRegister();
    // Set triton mode.
    ctx_.setMode(triton::modes::AST_OPTIMIZATIONS, true);
    ctx_.setMode(triton::modes::CONSTANT_FOLDING, true);
    // Initialize register
    ctx_.setConcreteRegisterValue(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_RIP), start_address);
    //ctx_.setConcreteRegisterValue(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_RSP), 0x500000);
    // Install a callback to lazy load memory accesses
    ctx_.addCallback(triton::callbacks::GET_CONCRETE_MEMORY_VALUE, OnLoadMemory);
    ctx_.addCallback(triton::callbacks::SET_CONCRETE_MEMORY_VALUE, OnStoreMemory);
    // TODO: Maybe we want to concretize the stack here? Initial value doesn't really matter...
}

void Emulator::SymbolizeAllRegister()
{
    // PC register
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_RIP), "rip");

    // Stack register
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_RSP), "rsp");

    // General purpose registers
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_RAX), "rax");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_RBX), "rbx");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_RCX), "rcx");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_RDX), "rdx");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_RSI), "rsi");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_RDI), "rdi");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_RBP), "rbp");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_R8), "r8");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_R9), "r9");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_R10), "r10");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_R11), "r11");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_R12), "r12");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_R13), "r13");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_R14), "r14");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_R15), "r15");

    // EFlags register
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_EFLAGS), "eflags");

    // Flag registers
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_CF), "cf");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZF), "zf");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_PF), "pf");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_OF), "of");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_AF), "af");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_SF), "sf");

    // Debug registers
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_DR0), "dr0");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_DR1), "dr1");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_DR2), "dr2");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_DR3), "dr3");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_DR6), "dr6");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_DR7), "dr7");

    // AVX512 registers
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM0), "zmm0");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM1), "zmm1");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM2), "zmm2");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM3), "zmm3");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM4), "zmm4");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM5), "zmm5");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM6), "zmm6");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM7), "zmm7");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM8), "zmm8");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM9), "zmm9");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM10), "zmm10");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM11), "zmm11");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM12), "zmm12");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM13), "zmm13");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM14), "zmm14");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM15), "zmm15");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM16), "zmm16");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM17), "zmm17");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM18), "zmm18");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM19), "zmm19");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM20), "zmm20");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM21), "zmm21");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM22), "zmm22");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM23), "zmm23");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM24), "zmm24");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM25), "zmm25");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM26), "zmm26");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM27), "zmm27");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM28), "zmm28");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM29), "zmm29");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM30), "zmm30");
    ctx_.symbolizeRegister(ctx_.getRegister(triton::arch::register_e::ID_REG_X86_ZMM31), "zmm31");

    // TODO: Add more registers here if required
}

void Emulator::ResetContext()
{
    ctx_.reset();
}

bool Emulator::EmulateOneInstruction(triton::arch::Instruction& instruction)
{
    if (ctx_.processing(instruction) == triton::arch::exception_e::NO_FAULT)
    {
        auto rip = ctx_.getRegisterAst(ctx_.getRegister(triton::arch::ID_REG_X86_RIP));
        if (rip->isSymbolized())
        {
            // RIP AST contains a symbolic variable, we don't know where to go...
            //Log("RIP is symbolized. Can't progress after {:#x}\n", instruction.GetAddress());
            return false;
        }
        return true;
    }
    return false; // Error happened during emulation, nontifies the caller.
}

bool Emulator::EmulateUntilSymbolic(ea_t start_address, std::shared_ptr<EmulatorHookInterface> hooks, size_t max_instructions_to_process)
{
    ea_t current = start_address;
    size_t number_of_instruction_processed = 0;
    while (number_of_instruction_processed < max_instructions_to_process)
    {
        auto val = FromAddress(current);
        if (!val.has_value())
            return false;
        triton::arch::Instruction instruction = val.value();
        ctx_.disassembly(instruction);
        if (hooks != nullptr)
            hooks->OnEmulateEnter(*this, instruction);
        // Run before hook if any
        if (ctx_.buildSemantics(instruction) != triton::arch::exception_e::NO_FAULT)
            return false; // Error happened
        // Run after hook, if any
        if (hooks != nullptr)
            hooks->OnEmulateExit(*this, instruction);

        auto rip = ctx_.getRegisterAst(ctx_.getRegister(triton::arch::ID_REG_X86_RIP));
        if (rip->isSymbolized())
        {
            // RIP AST contains a symbolic variable, we don't know where to go...
            //Log("RIP is symbolized. Can't progress after {:#x}\n", instruction.GetAddress());
            return true;
        }
        current = static_cast<uint64_t>(rip->evaluate());
        number_of_instruction_processed++;
    }
    return false; // Error happened during emulation, nontifies the caller.
}

std::unordered_map<uint64_t, triton::engines::symbolic::SharedSymbolicExpression>
Emulator::SliceExpression(const triton::engines::symbolic::SharedSymbolicExpression& expr)
{
    return ctx_.sliceExpressions(expr);
}

void Emulator::SymbolizeExpression(const triton::engines::symbolic::SharedSymbolicExpression& expr)
{
    ctx_.symbolizeExpression(expr->getId(), expr->getAst()->getBitvectorSize());
}

std::shared_ptr<llvm::Module> Emulator::ConvertToLLVM(const triton::ast::SharedAbstractNode node)
{
    llvm::LLVMContext llvm_context;
    triton::ast::TritonToLLVM lifter(llvm_context);
    std::shared_ptr<llvm::Module> llvm_module = lifter.convert(node);
    return llvm_module;
}

const triton::arch::Register& Emulator::GetRegister(triton::arch::register_e reg) const
{
    return ctx_.getRegister(reg);
}

std::optional<triton::uint512> Emulator::GetRegisterValue(triton::arch::register_e reg)
{
    const triton::arch::Register& r = ctx_.getRegister(reg);
    return GetRegisterValue(r);
}

std::optional<triton::uint512> Emulator::GetRegisterValue(const triton::arch::Register& reg)
{
    if (ctx_.isRegisterSymbolized(reg))
        return std::nullopt;

    return ctx_.getConcreteRegisterValue(reg);
}

void Emulator::SetRegisterValue(triton::arch::register_e reg, uint64_t value)
{
    auto& r = ctx_.getRegister(reg);
    ctx_.setConcreteRegisterValue(r, value, false);
}

triton::ast::SharedAbstractNode Emulator::GetRegisterAst(triton::arch::register_e reg)
{
    const triton::arch::Register& r = ctx_.getRegister(reg);
    return ctx_.getRegisterAst(r);
}

triton::ast::SharedAbstractNode Emulator::GetRegisterAst(const triton::arch::Register& reg)
{
    return ctx_.getRegisterAst(reg);
}

triton::ast::SharedAbstractNode Emulator::GetOperandAst(const triton::arch::OperandWrapper& op)
{
    return ctx_.getOperandAst(op);
}

triton::ast::SharedAbstractNode Emulator::GetMemoryAst(const triton::arch::MemoryAccess& mem)
{
    return ctx_.getMemoryAst(mem);
}

const triton::engines::symbolic::SharedSymbolicExpression& Emulator::GetSymbolicRegisterExpression(triton::arch::register_e reg)
{
    const triton::arch::Register& r = ctx_.getRegister(reg);
    return GetSymbolicRegisterExpression(r);
}

const triton::engines::symbolic::SharedSymbolicExpression& Emulator::GetSymbolicRegisterExpression(const triton::arch::Register& reg)
{
    return ctx_.getSymbolicRegister(reg);
}

const triton::engines::symbolic::SharedSymbolicExpression Emulator::GetSymbolicMemoryExpression(triton::arch::register_e reg)
{
    const triton::arch::Register& r = ctx_.getRegister(reg);
    uint64_t address = static_cast<uint64_t>(ctx_.getConcreteRegisterValue(r));
    return ctx_.getSymbolicMemory(address);
}

std::optional<triton::uint512> Emulator::GetMemoryValue(triton::arch::MemoryAccess& mem)
{
    if (!ctx_.isConcreteMemoryValueDefined(mem) || ctx_.isMemorySymbolized(mem))
        return std::nullopt;
    return ctx_.getConcreteMemoryValue(mem);
}

// FIXME: Returning by value??
std::optional<std::vector<uint8_t>> Emulator::GetMemoryValue(uint64_t address, size_t size)
{
    if (!ctx_.isConcreteMemoryValueDefined(address, size) || ctx_.isMemorySymbolized(address, size))
        return std::nullopt;
    return ctx_.getConcreteMemoryAreaValue(address, size, false);
}

std::optional<triton::uint512> Emulator::GetMemoryValue(triton::arch::register_e reg, size_t size)
{
    const triton::arch::Register& r = ctx_.getRegister(reg);
    const triton::ast::SharedAbstractNode register_ast = ctx_.getRegisterAst(r);
    if (register_ast->isSymbolized())
        return std::nullopt; // One symbolic variable is found in the expression. Can't find the address
    uint64_t address = static_cast<uint64_t>(register_ast->evaluate());
    if (!ctx_.isConcreteMemoryValueDefined(address, size) || ctx_.isMemorySymbolized(address, size))
        return std::nullopt;
    return ctx_.getConcreteMemoryValue(triton::arch::MemoryAccess(address, size));
}

triton::engines::symbolic::SharedSymbolicVariable Emulator::SymbolizeRegister(const triton::arch::Register& reg, const std::string& alias)
{
    return ctx_.symbolizeRegister(reg);
}

bool Emulator::IsSymbolic(const triton::arch::Register& reg) const
{
    return ctx_.isRegisterSymbolized(reg);
}

std::optional<triton::arch::Instruction> Emulator::FromAddress(ea_t address)
{
    insn_t insn;
    size_t size = decode_insn(&insn, address);
    if (size <= 0)
        return std::nullopt;
    char opcodes[16] = { 0 };
    get_bytes(opcodes, size, address);
    return triton::arch::Instruction(address, opcodes, size);
}

} // namespace hxemu