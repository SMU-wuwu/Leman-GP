"""FGM compute/dropout control: two equally weighted clean stochastic forwards."""
def two_clean_backward(wrapped,inputs,autocast,check_finite):
    with wrapped.no_sync():
        with autocast():first,_=wrapped(**inputs)
        check_finite(first,'first_clean');(.5*first).backward()
    with autocast():second,_=wrapped(**inputs)
    check_finite(second,'second_clean');(.5*second).backward()
    return .5*first.detach()+.5*second.detach(),{'clean1':float(first.detach()),'clean2':float(second.detach()),'perturbation_l2':0.,'two_stochastic_forwards':True}
