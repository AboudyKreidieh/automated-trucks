function  fcn(u)
    coder.extrinsic('bdroot')
    coder.extrinsic('set_param')
    if u > 0
        set_param(bdroot,'SimulationCommand','pause')
    end
end