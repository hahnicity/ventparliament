import matplotlib.pyplot as plt
from numpy import nan
import numpy as np
from numpy.linalg import lstsq


# this is the main function that performs the Howe method
def split_parameters(pressure,
                    flow,
                    volume,
                    sampling_frequency,
                    volumes_func,
                    prev_peep=nan,
                    end_insp=nan,
                    prev_P_max=nan,
                    ):
    Ei = nan
    Ri = nan
    Ee = nan
    Re = nan
    decay = nan

    # Start of inspiration:
    # This is at first crossing from negative flow to
    # positive flow, or at the first index. Stop looking
    # after max flow value. That point is definitely in
    # inspiration. Work backwards from max flow, because
    # start data can be noisy around zero crossing.
    start_insp = 0
    Q_max_index = np.argmax(flow[:len(flow)//3])
    i = Q_max_index
    while(i > 0.01):
        if(flow[i] >= 0.01 and flow[i-1] < 0.01):
            start_insp = i
            i = 0
        i -= 1

    # Get peep
    # From the average of last third of data
    peep_data = pressure[-len(pressure)//3:]
    peep = sum(peep_data)/len(peep_data)

    # End of inspiration:
    # Working backwards, find the last
    # point of positive flow in the data
#    end_insp = len(flow) - 1
#    i = len(flow)*1/2
#    while i > 0:
#        if(flow[i] < 0.01 and flow[i-1] > 0.01):
#            end_insp = i
#            i = 0
#        i -= 1
    if(np.isnan(end_insp)):
        end_insp = start_insp + 15
        i = end_insp
        while(i < len(flow) - 4):
            if((flow[i] < 0.01 or flow[i+1] < 0) and flow[i+2] < 0.0):
                end_insp = i
                i = len(flow)
            i += 1

    # start of expiration
    start_exp = end_insp + 1
#    i = len(flow)*1/2
#    while(i > end_insp):
#        if(flow[i] < 0 and flow[i-1] >= 0):
#            start_exp = i
#            i = 0
#        i -= 1

    # Get start and end point away from edges of insp
    # for parameter ID
    #start = start_insp + (end_insp - start_insp)*1/8
    start = start_insp + 5
    end = end_insp - 5

    # Remove peep from pressure
    # Offset for all pressure data is now 0
    # Offset of estimated pressure will be 0
    pressure = [p - peep for p in pressure]

    # Mess with pressure in expiration to make constant 0
    #k = start_exp
    #while k < len(pressure):
    #    pressure[k] = 0
    #    k += 1

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Check there are more than the minimum data points
    # needed for least squares in inspiration and range
    # for estimating data
    if (
      end_insp - start_insp <= 3    # Data too short
      or end - start <= 3           # Data too short
      or np.isnan(flow[0])          # Data starts with NaN
      or end_insp > len(flow) - 20  # End too close
      ):
        P_max = np.nan
    else:
        #------------------------------------------------------------------------
        #------------------------------------------------------------------------
        #------------------------------------------------------------------------
        # Params for INSP

        # Crop data to insp range
        flw = flow[start:end]
        pres = pressure[start:end]
        vol = volume[start:end]

        # Params from insp pressure
        dependent = np.array([pres])
        independent = np.array([flw, vol])

        try:
            res = lstsq(independent.T, dependent.T)
            Ei = res[0][1][0]
            Ri = res[0][0][0]
        except(ValueError):
            print('ValueError: Data has nan?')
            Ei = nan
            Ri = nan

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Params for EXP

        # Get P0 from end insp. That means the zero crossing pressure
        # If pressure dropped from max by 25%, use max pressure
        P_max= max(pressure[start_insp:start_exp])
        P_max_index = pressure.index(P_max)
        P_start = P_max

        # relative to start of expiration, so will be behind
        zero_crossing_offset = -(flow[start_exp])/(flow[start_exp - 1] - flow[start_exp])
        zero_crossing_pressure = ((pressure[start_exp-1] - pressure[start_exp])* zero_crossing_offset
                               + pressure[start_exp])
        P_start = zero_crossing_pressure

        # Shift reference points and crop to expiration only
        drop_pressure = [p - P_start for p in pressure[start_exp:]]
        drop_volume = [v - volume[start_exp] for v in volume[start_exp:]]
        drop_flow = [f for f in flow[start_exp:]]

        # DEFINE START AND END
        start = drop_flow.index(min(drop_flow)) + 5
        end = len(drop_volume) - 5

        dependent = np.array([drop_pressure[start:end]])
        independent = np.array([drop_volume[start:end], drop_flow[start:end]])
        Ee = nan
        Re = nan
        if(end - start > 5):
            try:
                expres = lstsq(independent.T, dependent.T)
                Ee = expres[0][0][0]
                Re = expres[0][1][0]
            except(ValueError):
                print('ValueError: Data has nan?')

        if(1):
            # fit for un-shifted data
            dependent = np.array([pressure[start_exp+start:end]])
            independent = np.array([volume[start_exp+start:end], flow[start_exp+start:end]])
            Ee_unshift = nan
            Re_unshift = nan
            if(end - start > 5) and independent.shape[1] > 0:
                try:
                    expres = lstsq(independent.T, dependent.T)
                    Ee_unshift = expres[0][0][0]
                    Re_unshift = expres[0][1][0]
                    # I removed fit error because it seems to be causing some
                    # corner case problems
                except(ValueError):
                    print('ValueError: Data has nan?')
            elif independent.shape[1] == 0:
                Ee_unshift = np.nan
                Re_unshift = np.nan
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Salwa method for better pressure estimation if end asynchrony
        b_offs = 2
        a_offs = 4
        search_offs = 20

        point_b = end_insp + b_offs
        point_a = point_b + a_offs
        point_b_pressure = max(pressure[point_b:point_b + search_offs])

        # pick steeper gradient of start and end of breath
        grad_end = (pressure[point_b] - pressure[point_a])
        grad_start = (pressure[start_insp+2] - pressure[start_insp+1])
        if(grad_start > grad_end/1.2):
            grad = grad_start
        else:
            grad = grad_end

        reconstruction_line = [min(P_max, grad*(b_offs + i) + point_b_pressure) for i in range(0, start_exp)]

        # area difference to choose the corner pressure
        area_difference = 0
        if(start_exp-P_max_index > 1):
            old_area = volumes_func(pressure[P_max_index:start_exp], 1/sampling_frequency)
            new_area = volumes_func(reconstruction_line[:start_exp-P_max_index], 1/sampling_frequency)
            area_difference = (new_area[-1] - old_area[-1]) / new_area[-1]

        # force reconstruction of inspiratory pressure for results
        if(area_difference > 1.525):
            start = start_insp + 8
            end = end_insp - 4
            #plt.plot(pressure)
            reconstruction_line.reverse()
            pressure = reconstruction_line
            #plt.plot(pressure)
            #plt.plot(start, pressure[start], 'o')
            #plt.plot(end, pressure[end], 'o')
            #plt.show()

            # Crop data to insp range
            pres = pressure[start:end]
            flw = flow[start:end]
            vol = volume[start:end]

            # Params from insp pressure
            dependent = np.array([pres])
            independent = np.array([flw, vol])

            try:
                res = lstsq(independent.T, dependent.T)
                Ei = res[0][1][0]
                Ri = res[0][0][0]
            except(ValueError):
                print('ValueError: Data has nan?')
                Ei = nan
                Ri = nan

    if(((abs(prev_peep - peep) < 0.5) or np.isnan(prev_peep))):
        return(Ei, Ri, Ee, Re, peep, P_max)
    else:
        return(nan, nan, nan, nan, peep, P_max)
