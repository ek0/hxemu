#include <gtest/gtest.h>

#include <hxemu.h>

#include <map>
#include <span>

// Demonstrate some basic assertions.
TEST(SimpleTest, BasicAssertions) {
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}

// Simple mock objects that does nothing. Used for simple initialization testing.
class SimpleInstructionBuilder : public hxemu::InstructionFactoryInterface
{
public:
    std::optional<triton::arch::Instruction> FromAddress(uint64_t address)
    {
        return std::nullopt;
    }

private:
    std::map<uint64_t, std::string> trace_;
};

TEST(EmulatorTest, Initialization)
{
    std::unique_ptr<SimpleInstructionBuilder> builder = std::make_unique<SimpleInstructionBuilder>();
    hxemu::Emulator emulator(triton::arch::ARCH_X86_64, std::move(builder));
}

TEST(EmulatorTest, EmulateOneInstruction)
{
    std::unique_ptr<SimpleInstructionBuilder> builder = std::make_unique<SimpleInstructionBuilder>();;
    hxemu::Emulator emulator(triton::arch::ARCH_X86_64, std::move(builder));
    char opcode[] = "\x48\xC7\xC0\x7B\x00\x00\x00"; // mov rax, 123
    triton::arch::Instruction instruction(opcode, sizeof(opcode));
    EXPECT_TRUE(emulator.EmulateOneInstruction(instruction));
    auto opt = emulator.GetRegisterValue(triton::arch::ID_REG_X86_RAX);
    EXPECT_TRUE(opt.has_value());
    EXPECT_EQ(opt.value(), 123);
}

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
    ctx.symbolizeMemory(mem);
}

// Simple mock objects that does nothing. Used for simple initialization testing.
class Trace : public hxemu::InstructionFactoryInterface
{
public:
    std::optional<triton::arch::Instruction> FromAddress(uint64_t address)
    {
        auto it = trace_.find(address);
        if (it == trace_.end())
            return std::nullopt;
        std::vector<uint8_t> s = it->second;
        auto instruction = triton::arch::Instruction(address, it->second.data(), it->second.size());
        return instruction;
    }

private:
    std::map<uint64_t, std::vector<uint8_t>> trace_ =
    {
        { 0 , { 0x48, 0xC7, 0xC0, 0x7B, 0x00, 0x00, 0x00} },
        { 7 , { 0xc3 } }
    };
};

TEST(EmulatorTest, RunUntilSymbolicRet)
{
    std::unique_ptr<Trace> builder = std::make_unique<Trace>();;
    hxemu::Emulator emulator(triton::arch::ARCH_X86_64, std::move(builder));
    emulator.SetMode(triton::modes::SYMBOLIZE_LOAD);
    emulator.SetOnLoadMemoryCallback(OnLoadMemory); // Properly prevent triton to concretize RIP during RET

    // Have to symbolize RSP.
    emulator.SymbolizeRegister(triton::arch::register_e::ID_REG_X86_RSP, "rsp");
    EXPECT_TRUE(emulator.Run(0));
    auto opt = emulator.GetRegisterValue(triton::arch::ID_REG_X86_RAX);
    EXPECT_TRUE(opt.has_value());
    EXPECT_EQ(opt.value(), 123);
}

// Simple mock objects that does nothing. Used for simple initialization testing.
class Trace2 : public hxemu::InstructionFactoryInterface
{
public:
    std::optional<triton::arch::Instruction> FromAddress(uint64_t address)
    {
        auto it = trace_.find(address);
        if (it == trace_.end())
            return std::nullopt;
        std::vector<uint8_t> s = it->second;
        auto instruction = triton::arch::Instruction(address, it->second.data(), it->second.size());
        return instruction;
    }

private:
    std::map<uint64_t, std::vector<uint8_t>> trace_ =
    {
        { 0 , { 0x48, 0x01, 0xCB } },
        { 3 , { 0x48, 0x29, 0xD3 } },
        { 6 , { 0x48, 0x8B, 0x03 } },
        { 9,  { 0xc3 }}
    };
};

TEST(EmulatorTest, GetSymbolicValue)
{
    std::unique_ptr<Trace2> builder = std::make_unique<Trace2>();;
    hxemu::Emulator emulator(triton::arch::ARCH_X86_64, std::move(builder));
    emulator.SetMode(triton::modes::SYMBOLIZE_LOAD);
    emulator.SetOnLoadMemoryCallback(OnLoadMemory); // Properly prevent triton to concretize RIP during RET

    // Have to symbolize RSP.
    emulator.SymbolizeRegister(triton::arch::register_e::ID_REG_X86_RSP, "rsp");
    EXPECT_TRUE(emulator.Run(0));
    EXPECT_TRUE(emulator.IsSymbolic(triton::arch::register_e::ID_REG_X86_RAX));
    auto opt = emulator.GetRegisterValue(triton::arch::ID_REG_X86_RAX);
    EXPECT_FALSE(opt.has_value());
}

TEST(EmulatorTest, TestLLVMLifting)
{
    std::unique_ptr<Trace2> builder = std::make_unique<Trace2>();;
    hxemu::Emulator emulator(triton::arch::ARCH_X86_64, std::move(builder));
    emulator.SetRepresentationMode(triton::ast::representations::mode_e::PCODE_REPRESENTATION);
    emulator.SetOnLoadMemoryCallback(OnLoadMemory); // Properly prevent triton to concretize RIP during RET

    // Have to symbolize RSP.
    emulator.SymbolizeAllRegister();
    EXPECT_TRUE(emulator.Run(0));
    EXPECT_TRUE(emulator.IsSymbolic(triton::arch::register_e::ID_REG_X86_RAX));
    auto opt = emulator.GetRegisterValue(triton::arch::ID_REG_X86_RAX);
    EXPECT_FALSE(opt.has_value());
    triton::ast::SharedAbstractNode node = emulator.GetRegisterAst(triton::arch::ID_REG_X86_RAX);
    EXPECT_NE(node, nullptr);
    std::shared_ptr<llvm::Module> mod = emulator.ConvertToLLVM(node);
    EXPECT_NE(mod, nullptr);
}