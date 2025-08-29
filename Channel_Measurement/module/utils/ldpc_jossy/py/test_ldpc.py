import pytest
import numpy as np
import ldpc

@pytest.mark.parametrize("standard,rate,z,ptype", [
    ("802.16","1/2",3,"A"),
    ("802.16","2/3",3,"A"),
    ("802.16","2/3",3,"B"),
    ("802.16","3/4",3,"A"),
    ("802.16","3/4",3,"B"),
    ("802.16","5/6",3,"A"),
    ("802.16","1/2",27,"A"),
    ("802.16","2/3",27,"A"),
    ("802.16","2/3",27,"B"),
    ("802.16","3/4",27,"A"),
    ("802.16","3/4",27,"B"),
    ("802.16","5/6",27,"A"),
    ("802.16","1/2",54,"A"),
    ("802.16","2/3",54,"A"),
    ("802.16","2/3",54,"B"),
    ("802.16","3/4",54,"A"),
    ("802.16","3/4",54,"B"),
    ("802.16","5/6",54,"A"),
    ("802.16","1/2",81,"A"),
    ("802.16","2/3",81,"A"),
    ("802.16","2/3",81,"B"),
    ("802.16","3/4",81,"A"),
    ("802.16","3/4",81,"B"),
    ("802.16","5/6",81,"A"),
    ("802.11n","1/2",27,"A"),
    ("802.11n","2/3",27,"A"),
    ("802.11n","3/4",27,"A"),
    ("802.11n","5/6",27,"A"),
    ("802.11n","1/2",54,"A"),
    ("802.11n","2/3",54,"A"),
    ("802.11n","3/4",54,"A"),
    ("802.11n","5/6",54,"A"),
    ("802.11n","1/2",81,"A"),
    ("802.11n","2/3",81,"A"),
    ("802.11n","3/4",81,"A"),
    ("802.11n","5/6",81,"A"),
])

def test_ldpc(standard, rate, z, ptype):
    print(standard,rate,z,ptype)
    mycode = ldpc.code(standard, rate, z, ptype)
    assert len(mycode.proto[0]) == 24
    H = mycode.pcmat()
    vdeg = mycode.vdeg
    cdeg = mycode.cdeg
    intrlv = mycode.intrlv
    assert np.sum(vdeg) == np.sum(cdeg)
    assert np.sum(vdeg) == np.sum(H)
    assert np.sum(vdeg) == len(intrlv)
    K = mycode.K
    for k in range(100):
        u = np.random.randint(0,2,K)
        x = mycode.encode(u)
        assert np.count_nonzero(np.mod(np.dot(x,np.transpose(H)),2)) == 0
        # modulate and amplify
        y = np.array(10*(.5-x), dtype=float)
        app,it = mycode.decode(y,'sumprod')
        assert it == 0
        xh = np.array(app<0, dtype=int)
        assert np.count_nonzero(xh != x) == 0
        app,it = mycode.decode(y,'sumprod2')
        assert it == 0
        xh = np.array(app<0, dtype=int)
        assert np.count_nonzero(xh != x) == 0
#        (app,it) = mycode.decode(y,'minsum')
#        assert it == 0
#        xh = np.array(app<0, dtype=int)
#        assert np.count_nonzero(xh != x) == 0
        
        

