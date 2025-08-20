**This directory includes some of our results.**  
**And here are a brief introduction**  

1. first group:  
    *ofdm_start_standard.png  
    ofdm_start_early10.png  
    ofdm_start_late10.png  
    received_symbol.png*  


    first three is constellation distribution.  
    'standard' means use the correlation peak to synchronize, 
    and 'early', 'late' stands for n sampling points before or after the correlation
    peak (where n in this e.g. is 10)  
    another thing you can see in pics is that constellations in last two maps are entirely
    distorted, which is caused by the amp declined in time domain (hardware problem, to avoid this, 
    we add white noise after signal in latter experiment)

2. second group:  
*amp filtered phase shift.png  
Amplitude of phase shift.png  
filtered_unwrap_data.png  
filtered_unwrap.png  
unwrap.png*  


    2 amps are raw amplitude of phase shift factor and filtered amplitude of that respectively, 
    filtered_data is phase of phase shift factor after filtering, and filtered_unwrap show the fitting res,
    noticed that there exists a big gap between postive freq and negative freq, we use only the positive freq 
    to fit the line in 'truncated' method, which is shown in unwrap.png
  
3. third group:  
*constellation_map.png  
constellation_raw.png  
constellation_corrected.png*  


    constellation_map.png is only extension exp of first group. And the rest two are raw 
    and corrected constellations respectively (corrected by calculated delta)

4. 4th group:
*impluse_response.png*


    the impluse reponse we measured, nothing special
