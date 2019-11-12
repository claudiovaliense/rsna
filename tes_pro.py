def prob_any(epi, intrap, intraven, subar, subdur):
    z_um = epi + intrap + intraven + subar + subdur
    z_dois = (epi*intrap) + (epi*intraven) + (epi*subar) + (epi*subdur) + (intrap*intraven) + (intrap*subar) + (intrap*subdur) + (intraven*subar) + (intraven*subdur) + (subar*subdur)
    z_tres = (epi*intrap*intraven) + (epi*intrap*subar) + (epi*intrap*subdur) + (epi*intraven*subar) + (epi*intraven*subdur) + (epi*subar*subdur) + (intrap*intraven*subar) + (intrap*intraven*subdur) + (intrap*subar*subdur) + (intraven*subar*subdur)
    z_quat = (epi*intrap*intraven*subar) + (epi*intrap*intraven*subdur) + (epi*intrap*subar*subdur) + (epi*intraven*subar*subdur) + (intrap*intraven*subar*subdur)
    z_cinc = epi*intrap*intraven*subar*subdur
    #print(z_cinc)
    pany = z_um - z_dois + z_tres - z_quat + z_cinc
    return(pany)

print(prob_any(0, 0, 0, 0, 0))
