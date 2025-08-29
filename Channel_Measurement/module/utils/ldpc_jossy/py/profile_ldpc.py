import cProfile
import ldpc_awgn

cProfile.run('ldpc_awgn.sim("802.11n","1/2",27,"A", 2,3.0)')
