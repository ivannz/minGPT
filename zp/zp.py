import os
import tqdm
import wikia
from wikia.wikia import WikiaError

from urllib.parse import unquote

use_transcript = False

out_file, in_file = 'zp.txt', 'zp_episodes.txt'
with open(in_file, 'tr') as fin:
    episodes = set([unquote(line) for line in map(str.strip, fin) if line])

done, ready_file = set(), 'zp_done.txt'
if os.path.isfile(ready_file):
    done = set(map(lambda x: x.strip(), open(ready_file)))

episodes -= done
with open(out_file, 'ta+') as fout, open(ready_file, 'at+') as ready:
    for episode in tqdm.tqdm(episodes):
        try:
            page = wikia.page('zeropunctuation', episode)
            if use_transcript:
                transcript = page.section('Transcript')

            else:
                transcript = page.content

            fout.write(f'EPISODE: {page.title}\n\n')
            if transcript is not None:
                fout.write(transcript.strip() + '\n\n\n')

        except WikiaError:
            print(f'could not read `{episode}`')

        else:
            ready.write(f'{episode}\n')

    # finally:
    #     open(ready_file, 'wt').writelines(map("{}\n".format, done))
