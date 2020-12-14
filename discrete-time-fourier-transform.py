#! /usr/local/bin/python3.8

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

plt.style.use('classic')
plt.style.use('seaborn-poster')

title_font = {'fontname' : 'DejaVu Serif'}

################################################################################

def discrete_time_fourier_transform(time_domain_sequence):
	'''
	Calculate the discrete-time Fourier transform (DTFT) of the input sequence.
	In theory, the DTFT of a sequence is continuous. Which cannot be represented by a computer.
	Hence, compute the discrete Fourier transform (DFT) of the input sequence--with a small change.
	In case of the DFT, the output sequence has the same number of elements as the input.
	But the DTFT should have infinitely many elements (because it is continuous).
	So, if the number of output sequence points is increased indefinitely, DFT becomes DTFT.
	It is reflected in the value of the variable 'n_points'.
	Note! DFT is the name of the process which is done by fast Fourier transform (FFT) algorithms.
	You may use 'DFT' and 'FFT' interchangeably.

	Args:
		time_domain_sequence: discrete sequence to be transformed

	Returns:
		DTFT of input sequence
	'''

	# initialise 'omega', the variable of frequency domain
	# in theory, for DTFT, 'omega' is a continuous variable assumed to take values from from -3.14 to 3.14
	# which is why 'n_points' should be large (to imitate continuous variable behaviour)
	# if 'n_points' equals 'len(time_domain_sequence)', the final output will be same as FFT
	# because the number of elements in the input and output sequences will be the same
	n_points = 1000
	omega = np.linspace(-np.pi, np.pi, n_points)

	# twiddle factor matrix
	# in case of FFT, it is a square matrix because 'n_points' equals 'len(time_domain_sequence)'
	# not in this case, though
	# DTFT is calculated by a matrix multiplication, just like DFT matrix multiplication
	twiddle = [[np.exp(-1j * omega * n) for n in range(len(time_domain_sequence))] for omega in np.linspace(-np.pi, np.pi, n_points)]
	frequency_domain_signal = np.array(twiddle) @ np.array(time_domain_sequence)

	return omega, frequency_domain_signal

################################################################################

if __name__ == '__main__':

	sampling_rate = 1800 # if it is not known, set this to `None'
	t = np.arange(0, 5, 1 / sampling_rate)
	normalise = True # whether frequency magnitude should be normalised to the range from -1 to 1

	# input sequence (values be greater than or equal to -1 but less than or equal to 1)
	# time_domain_sequence = [0.3415063509461096, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604] # Db4 low-pass
	# time_domain_sequence = [-0.1294095225512604, -0.2241438680420134, 0.8365163037378079, -0.3415063509461096] # Db4 high-pass
	# time_domain_sequence = [0.25, 0.75, 0.75, 0.25] # biorthogonal low-pass
	# time_domain_sequence = [-0.25, -0.75, 0.75, 0.25] # biorthogonal high-pass
	# time_domain_sequence = np.sin(2 * np.pi * 20 * np.arange(0, 1, 1 / sampling_rate))
	time_domain_sequence = np.cos(2 * np.pi * 1000 * t)

	# output signal
	omega, frequency_domain_signal = discrete_time_fourier_transform(time_domain_sequence)
	if normalise:
		frequency_domain_signal /= np.amax(np.abs(frequency_domain_signal))

	########################################

	# open a window to display sequence
	fig = plt.figure()
	fig.canvas.set_window_title('Discrete Fourier Transform')
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(time_domain_sequence, color = 'red', linewidth = 0.8)

	# appearance settings
	h_length = len(time_domain_sequence)
	ax.axhline(linewidth = 1.6, color = 'k')
	ax.axvline(linewidth = 1.6, color = 'k')
	ax.grid(True, which = 'major', linewidth = 0.6)
	ax.grid(True, which = 'minor', linewidth = 0.3)
	ax.minorticks_on()
	ax.set_xlim(0, h_length - 1)
	ax.set_ylim(-1, 1)
	ax.set_yticks(np.linspace(-1, 1, 11))
	fig.canvas.draw()

	# fill text
	ax.set_title('Time Domain', **title_font)
	ax.set_xlabel(r'$n$')
	ax.set_ylabel(r'$x[n]$')
	ax.set_xticklabels([r'${}$'.format(t.get_text()) for t in ax.get_xticklabels()])
	ax.set_yticklabels([r'${}$'.format(t.get_text()) for t in ax.get_yticklabels()])

	fig.tight_layout(pad = 1)

	########################################

	# open a window to display signal magnitude
	fig = plt.figure()
	fig.canvas.set_window_title('Discrete Fourier Transform')
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(omega, np.abs(frequency_domain_signal), 'r-', linewidth = 0.8)

	# appearance settings
	ax.axhline(linewidth = 1.6, color = 'k')
	ax.axvline(linewidth = 1.6, color = 'k')
	ax.grid(True, which = 'major', linewidth = 0.6)
	ax.grid(True, which = 'minor', linewidth = 0.3)
	ax.minorticks_on()
	ax.set_xlim(-np.pi, np.pi)
	if normalise:
		ax.set_yticks(np.linspace(0, 1, 11))
	fig.canvas.draw()

	# fill text
	ax.set_title('Frequency Domain: Magnitude Plot', **title_font)
	if sampling_rate is None:
		ax.set_xticklabels([r'$-\pi$', r'$-\dfrac{3\pi}{4}$', r'$-\dfrac{\pi}{2}$', r'$-\dfrac{\pi}{4}$', r'$0$', r'$\dfrac{\pi}{4}$', r'$\dfrac{\pi}{2}$', r'$\dfrac{3\pi}{4}$', r'$\pi$'])
		ax.set_xticks([i * np.pi / 4 for i in range(-4, 5)])
		ax.set_xlabel(r'$\omega$')
	else:
		ax.set_xticklabels([f'${i * sampling_rate / 20:.2f}$' for i in range(-10, 11)])
		ax.set_xticks([i * np.pi / 10 for i in range(-10, 11)])
		ax.set_xlabel(r'$f/$Hz', **title_font)
	ax.set_ylabel(r'$|X(e^{j\omega})|$')
	ax.set_yticklabels([r'${}$'.format(t.get_text()) for t in ax.get_yticklabels()])

	fig.tight_layout(pad = 1)

	########################################

	# open a window to display signal phase
	fig = plt.figure()
	fig.canvas.set_window_title('Discrete Fourier Transform')
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(omega, np.angle(frequency_domain_signal), 'r-', linewidth = 0.8)

	# appearance settings
	ax.axhline(linewidth = 1.6, color = 'k')
	ax.axvline(linewidth = 1.6, color = 'k')
	ax.grid(True, which = 'major', linewidth = 0.6)
	ax.grid(True, which = 'minor', linewidth = 0.3)
	ax.minorticks_on()
	ax.set_xlim(-np.pi, np.pi)
	ax.set_ylim(-np.pi, np.pi)
	fig.canvas.draw()

	# fill text
	ax.set_title('Frequency Domain: Phase Plot', **title_font)
	if sampling_rate is None:
		ax.set_xticklabels([r'$-\pi$', r'$-\dfrac{3\pi}{4}$', r'$-\dfrac{\pi}{2}$', r'$-\dfrac{\pi}{4}$', r'$0$', r'$\dfrac{\pi}{4}$', r'$\dfrac{\pi}{2}$', r'$\dfrac{3\pi}{4}$', r'$\pi$'])
		ax.set_xticks([i * np.pi / 4 for i in range(-4, 5)])
		ax.set_xlabel(r'$\omega$')
	else:
		ax.set_xticklabels([f'${i * sampling_rate / 20:.2f}$' for i in range(-10, 11)])
		ax.set_xticks([i * np.pi / 10 for i in range(-10, 11)])
		ax.set_xlabel(r'$f/$Hz', **title_font)
	ax.set_yticklabels([r'$-\pi$', r'$-\dfrac{3\pi}{4}$', r'$-\dfrac{\pi}{2}$', r'$-\dfrac{\pi}{4}$', r'$0$', r'$\dfrac{\pi}{4}$', r'$\dfrac{\pi}{2}$', r'$\dfrac{3\pi}{4}$', r'$\pi$'])
	ax.set_yticks([i * np.pi / 4 for i in range(-4, 5)])

	fig.tight_layout(pad = 1)

	plt.show()

