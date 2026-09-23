#ifndef PARLAY_INTERNAL_FCONTEXT_H_
#define PARLAY_INTERNAL_FCONTEXT_H_

// The context switch, vendored from Boost.Context so that the library stays
// header-only and does not depend on <boost/context/detail/fcontext.hpp>,
// which Boost does not treat as public API.
//
// Source: Boost 1.83.0, boostorg/context tag boost-1.83.0, src/asm/
//   jump_x86_64_sysv_elf_gas.S   sha256 650df99d0198ff8e...
//   make_x86_64_sysv_elf_gas.S   sha256 d79127715565ecab...
// Copyright Oliver Kowalke 2009.  Distributed under the Boost Software
// License, Version 1.0; see fcontext.LICENSE_1_0.txt next to this file or
// http://www.boost.org/LICENSE_1_0.txt.
//
// Changes from the originals:
//   - Only the x86-64 System V LP64 configuration is kept.  The conditional
//     paths for CET shadow stacks, TSX, the TLS stack protector and ILP32 are
//     resolved away.  MXCSR and the x87 control word are still saved and
//     restored, as the System V ABI makes them callee-saved.
//   - Symbols are renamed spork_jump_fcontext / spork_make_fcontext so they
//     cannot collide with a Boost.Context linked into the same process.
//   - The code lives in COMDAT groups inside file-scope inline assembly, so
//     every translation unit that includes this header emits it and the
//     linker keeps one copy.  Internal labels are assembler-local.
//   - endbr64 marks each entry point.  It is a NOP without CET enforcement.
//
// Frame layout at the context-data pointer (what an fcontext_t points at):
//   0x00 MXCSR  0x04 x87 CW  0x08 (unused)  0x10 R12  0x18 R13  0x20 R14
//   0x28 R15    0x30 RBX     0x38 RBP       0x40 RIP
//
// A handle to a stack is produced by the jump that leaves it (RAX = the RSP
// where these registers were pushed), so a handle exists only for a stack
// nobody is running on.  The handoff protocol in spork_scheduler.h relies on
// that property.

#include <cstddef>

namespace spork {

using fcontext_t = void*;

// Returned by a jump, and passed to a context function on entry: the handle
// of whichever stack jumped here, and the data word it sent.
struct transfer_t {
  fcontext_t fctx;
  void* data;
};

extern "C" transfer_t spork_jump_fcontext(fcontext_t to, void* vp);
extern "C" fcontext_t spork_make_fcontext(void* sp, std::size_t size, void (*fn)(transfer_t));

// Switch to `to`, sending `vp`; returns when some context jumps back here.
inline transfer_t jump_fcontext(fcontext_t to, void* vp) {
  return spork_jump_fcontext(to, vp);
}

// Prepare a fresh stack whose top is `sp` so that the first jump to the
// returned handle enters `fn` on it.  `fn` must never return.
inline fcontext_t make_fcontext(void* sp, std::size_t size, void (*fn)(transfer_t)) {
  return spork_make_fcontext(sp, size, fn);
}

}  // namespace spork

#if !defined(__x86_64__) || !defined(__ELF__) || defined(_ILP32)
#error "fcontext.h: only x86-64 System V LP64 ELF is supported"
#endif

__asm__(R"asm(
    .section .text.spork_jump_fcontext,"axG",@progbits,spork_jump_fcontext,comdat
    .globl spork_jump_fcontext
    .type spork_jump_fcontext,@function
    .align 16
spork_jump_fcontext:
    endbr64
    leaq  -0x40(%rsp), %rsp     /* prepare stack */
    stmxcsr  (%rsp)             /* save MMX control- and status-word */
    fnstcw   0x4(%rsp)          /* save x87 control-word */
    movq  %r12, 0x10(%rsp)      /* save R12 */
    movq  %r13, 0x18(%rsp)      /* save R13 */
    movq  %r14, 0x20(%rsp)      /* save R14 */
    movq  %r15, 0x28(%rsp)      /* save R15 */
    movq  %rbx, 0x30(%rsp)      /* save RBX */
    movq  %rbp, 0x38(%rsp)      /* save RBP */
    movq  %rsp, %rax            /* store RSP (pointing to context-data) in RAX */
    movq  %rdi, %rsp            /* restore RSP (pointing to context-data) from RDI */
    movq  0x40(%rsp), %r8       /* restore return-address */
    ldmxcsr  (%rsp)             /* restore MMX control- and status-word */
    fldcw    0x4(%rsp)          /* restore x87 control-word */
    movq  0x10(%rsp), %r12      /* restore R12 */
    movq  0x18(%rsp), %r13      /* restore R13 */
    movq  0x20(%rsp), %r14      /* restore R14 */
    movq  0x28(%rsp), %r15      /* restore R15 */
    movq  0x30(%rsp), %rbx      /* restore RBX */
    movq  0x38(%rsp), %rbp      /* restore RBP */
    leaq  0x48(%rsp), %rsp      /* prepare stack */
    movq  %rsi, %rdx            /* return transfer_t: RAX == fctx, RDX == data */
    movq  %rax, %rdi            /* pass transfer_t as first arg: RDI == fctx, RSI == data */
    jmp  *%r8                   /* indirect jump to context */
    .size spork_jump_fcontext,.-spork_jump_fcontext

    .section .text.spork_make_fcontext,"axG",@progbits,spork_make_fcontext,comdat
    .globl spork_make_fcontext
    .type spork_make_fcontext,@function
    .align 16
spork_make_fcontext:
    endbr64
    movq  %rdi, %rax            /* first arg == top of context-stack */
    andq  $-16, %rax            /* shift address in RAX to lower 16 byte boundary */
    leaq  -0x48(%rax), %rax     /* reserve space for context-data on context-stack;
                                   on context-function entry: (RSP - 0x8) % 16 == 0 */
    movq  %rdx, 0x30(%rax)      /* third arg == address of context-function, stored in RBX */
    stmxcsr  (%rax)             /* save MMX control- and status-word */
    fnstcw   0x4(%rax)          /* save x87 control-word */
    leaq  .Lspork_fc_trampoline(%rip), %rcx
    movq  %rcx, 0x40(%rax)      /* return-address for the first jump: trampoline */
    leaq  .Lspork_fc_finish(%rip), %rcx
    movq  %rcx, 0x38(%rax)      /* return-address for the context-function: finish */
    ret                         /* return pointer to context-data */
.Lspork_fc_trampoline:
    endbr64
    push %rbp                   /* store return address on stack; fix stack alignment */
    jmp *%rbx                   /* jump to context-function */
.Lspork_fc_finish:
    endbr64
    xorq  %rdi, %rdi            /* exit code is zero */
    call  _exit@PLT             /* exit application */
    hlt
    .size spork_make_fcontext,.-spork_make_fcontext
    .text
)asm");

#endif  // PARLAY_INTERNAL_FCONTEXT_H_
