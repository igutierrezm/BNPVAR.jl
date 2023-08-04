@testset "gen_M()" begin
    N = 2
    p = 3
    @test BV.gen_M(N, p) == N * (1 + N * p)
end

@testset "init_Γ()" begin
    N = 2
    @test BV.init_Γ(N) == ones(Bool, 2, 2)
end

@testset "init_A()" begin
    N = 2
    K = 2
    p = 3
    A = BV.init_A(N, K, p)
    @test length(A) == K
    for k in 1:K
        @test length(A[k]) == p
        for m in 1:p
            @test size(A[k][m]) == (N, N)
            for i in 1:N, j in 1:N
                if i == j
                    @test abs(A[k][m][i, j]) < 1
                else
                    @test A[k][m][i, j] == 0
                end
            end
        end
    end
end

@testset "init_c()" begin
    N = 2
    K = 2
    c = BV.init_c(N, K)
    @test length(c) == K
    for k in 1:K
        @test length(c[k]) == N
        for i in 1:N
            @test abs(c[k][i]) < 2
        end
    end
end

@testset "init_Σ()" begin
    N = 2
    K = 2
    Σ = BV.init_Σ(N, K)
    @test length(Σ) == K
    for k in 1:K
        for i in 1:N, j in 1:N
            if i == j
                @test Σ[k][i, j] > 0
            else
                @test Σ[k][i, j] == 0
            end
        end
    end
end

@testset "gen_gdict(): Example 1" begin
    N = 1
    p = 2
    gdict = BV.gen_gdict(N, p)
    @test length(gdict) == N^2
    @test gdict[1] == Bool.([0, 1, 1])
end

@testset "gen_gdict(): Example 2" begin
    N = 2
    p = 1
    gdict = BV.gen_gdict(N, p)
    B = [0 1 3; 0 2 4]' |> vec
    @test length(gdict) == N^2
    @test gdict[1] == (B .== 1)
    @test gdict[2] == (B .== 2)
end

@testset "gen_gdict(): Example 3" begin
    N = 2
    p = 2
    gdict = BV.gen_gdict(N, p)
    B = [0 1 3 1 3; 0 2 4 2 4]' |> vec
    @test length(gdict) == N^2
    @test gdict[1] == (B .== 1)
    @test gdict[2] == (B .== 2)
end

@testset "gen_yvec()" begin
    N = 3
    T = 3
    p = 1
    Y = ones(T, N)
    Y[:] = 1:(T * N)
    yvec = BV.gen_yvec(Y, T, p)
    @test yvec[1] == [2.0, 5.0, 8.0]
    @test yvec[2] == [3.0, 6.0, 9.0]
end

@testset "gen_Z()" begin
    N = 3
    T = 4
    p = 2
    Y = ones(T, N)
    Y[:] = 1:(T * N)
    Z = BV.gen_Z(Y, T, p)
    @test size(Z) == (T - p, 1 + N * p)
    @test Z[1, :] == [1.0, 2.0, 6.0, 10.0, 1.0, 5.0, 09.0]
    @test Z[2, :] == [1.0, 3.0, 7.0, 11.0, 2.0, 6.0, 10.0]
end

@testset "gen_Xvec()" begin
    N = 3
    T = 4
    p = 2
    Y = ones(T, N)
    Y[:] = 1:(T * N)
    Xvec = BV.gen_Xvec(Y, N, T, p)
    @test length(Xvec) == T - p
    @test Xvec[1] == kron(I(N), [1.0, 2.0, 6.0, 10.0, 1.0, 5.0, 09.0]')
    @test Xvec[2] == kron(I(N), [1.0, 3.0, 7.0, 11.0, 2.0, 6.0, 10.0]')
end

@testset "gen_y()" begin
    N = 2
    T = 3
    p = 1
    Y = ones(T, N)
    Y[:] = 1:(T * N)
    y = BV.gen_yvec(Y, T, p) |> BV.gen_y
    @test y == [2.0, 5.0, 3.0, 6.0]
end

@testset "gen_X()" begin
    N = 3
    T = 4
    p = 2
    Y = ones(T, N)
    Y[:] = 1:(T * N)
    X = BV.gen_Xvec(Y, N, T, p) |> BV.gen_X
    @test size(X, 2) == N * (1 + N * p)
    @test size(X, 1) == N * (T - p)
    @test X[1, :]' == kron([1 0 0], [1.0 2.0 6.0 10.0 1.0 5.0 09.0])
    @test X[2, :]' == kron([0 1 0], [1.0 2.0 6.0 10.0 1.0 5.0 09.0])
    @test X[4, :]' == kron([1 0 0], [1.0, 3.0, 7.0, 11.0, 2.0, 6.0, 10.0]')
    @test X[5, :]' == kron([0 1 0], [1.0, 3.0, 7.0, 11.0, 2.0, 6.0, 10.0]')
end

@testset "gen_B()" begin
    N = 2
    K = 1
    p = 2
    c = BV.init_c(N, K)
    A = BV.init_A(N, K, p)
    B = BV.gen_B(c, A, K)
    @test B[1][1, :] == c[1]
    @test B[1][2:3, :] == A[1][1]'
    @test B[1][4:5, :] == A[1][2]'
end

@testset "gen_B()" begin
    c = [[0, 0]]
    A = [[[1 3; 2 4]]]
    β = BV.gen_β(c, A, 1)
    @test β[1] == [0, 1, 3, 0, 2, 4]
end

@testset "gen_g()" begin
    g = Bool.([1 0; 1 1]) |> BV.gen_g
    @test g == [1, 1, 0, 1]
end

@testset "gen_ψ()" begin
    p = 1
    N = 2
    g = [1, 1, 0, 1]
    M = BV.gen_M(N, p)
    gdict = BV.gen_gdict(N, p)
    ψ = BV.gen_ψ(g, gdict, N, M)
    @show g
    @show gdict
    @show ψ
    @test ψ == [1 1 0; 1 1 1]' |> vec
end
