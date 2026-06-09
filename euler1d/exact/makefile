#	Makefile  1.01       July. 27, 1996
#F90 = ifx
F90 = gfortran

#F90FLAGS = -r8 -fpe0 -traceback -g #-check all -warn unused -g
F90FLAGS = -std=legacy -ffpe-trap=invalid,zero,overflow -fdec -fdefault-real-8 -fbacktrace -g #-check uninit,bound #,stack,arg,conti

EXEC = e1rpex
WORKOBJS = e1rpex.o

$(EXEC) : $(WORKOBJS)
	$(F90) $(WORKOBJS) -o $(EXEC)

.f90.o:
	$(F90) $(F90FLAGS) -c $<

clean:
	rm *.o

.SUFFIXES: .f90
