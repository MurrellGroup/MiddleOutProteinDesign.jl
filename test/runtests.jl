using Test
using MiddleOutProteinDesign
using BranchingFlows

@testset "MiddleOutProteinDesign" begin
    @test MiddleOutProteinDesign.P_flowception isa DirectionalFlowceptionFlow
    @test MiddleOutProteinDesign.flowception_reveal_temperature isa Float32
    tuned = MiddleOutProteinDesign.with_reveal_temperature(MiddleOutProteinDesign.P_flowception, 10f0)
    @test tuned.total_time == MiddleOutProteinDesign.P_flowception.total_time
    @test tuned.reveal_order isa SeededRevealOrder
    @test tuned.reveal_order.temperature == 10f0
end
