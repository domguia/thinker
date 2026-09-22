# Qualitative comparison: Thinker vs checkpoints/retrieval1_embedkd_wsd_best.pt

- Thinker checkpoint: `checkpoints/retrieval1_ab_topkonly_best.pt`
- Second Thinker checkpoint: `checkpoints/retrieval1_embedkd_wsd_best.pt`
- Sample: first 30 rows of `data/distill/hotpotqa_full/val.jsonl` (fixed, deterministic)
- Decoding: temperature=0.8, top_p=0.9, seed=0

## Example 0 (num_hops=2)
**Question**: What is the approximate capacity of passengers per year of the airport that is considered Condor Flugdienst's main base?
**Gold answer**: 65 million
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>'

## Example 1 (num_hops=2)
**Question**: What Japanese mixed martial artist and professional wrestler fought a former UFC Heavyweight Champion who was previously associated with Mark Coleman's Team Hammer House?
**Gold answer**: Kenichi Yamamoto
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think> Am 1'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think> Carr E'

## Example 2 (num_hops=2)
**Question**: What is the current home stadium for the California football team which formerly included John David Crow?
**Gold answer**: Levi's Stadium
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think> missionariesvej'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>veism.'

## Example 3 (num_hops=2)
**Question**: In what city does Mike Carey coach women's basketball ?
**Gold answer**: Morgantown
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>, New.'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think> of Kingdom of'

## Example 4 (num_hops=2)
**Question**: Did Bedtime Stories and Big Red both star Walter Pidgeon?
**Gold answer**: no
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: 'yes'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: 'yes'

## Example 5 (num_hops=2)
**Question**: Which breed of dog, Brazilian Dogo or Tahltan Bear Dog, is a Molosser-type working dog breed?
**Gold answer**: Brazilian Dogo
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think> B'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think> the'

## Example 6 (num_hops=2)
**Question**: Omar Metwally played "Dr. Fahim Nasir" in a movie directed by who?
**Gold answer**: Jaume Collet-Serra
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>field'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>ovich'

## Example 7 (num_hops=2)
**Question**: "Shot in the Dark" is a single from an album released by which record label ?
**Gold answer**: Roadrunner Records
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>ay River'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>it "'

## Example 8 (num_hops=2)
**Question**: The actress that played Beth Jordache in the soap "Brookside" also starred in a British film with Oliver Milburn that was directed by who?
**Gold answer**: Sandra Goldbacher
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>ther'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>ap'

## Example 9 (num_hops=2)
**Question**: In which city was this English new wave and synth-pop band that released the studio album "Liberty" formed?
**Gold answer**: Birmingham
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>'

## Example 10 (num_hops=2)
**Question**: Which American rapper produced an album containing a song that was nominated for a Grammy Award?
**Gold answer**: 2Pac
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think> 27, 1922996899990997'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>'

## Example 11 (num_hops=2)
**Question**: No. 11 Squadron RAAF was based at what base, 25 km north of Adelaide?
**Gold answer**: RAAF Base Edinburgh
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think> Patrick World'

## Example 12 (num_hops=2)
**Question**: Guilermo Del Toro directed Doug Jones in a 1997 horror film based on whose short story?
**Gold answer**: Donald A. Wollheim
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>ilton L 2'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think> District I.ian'

## Example 13 (num_hops=2)
**Question**: When was the release date of the American fantasy comedy family film that was directed and wrote by Sven Davison and David Dobkin
**Gold answer**: The film was released on November 9, 2007 in the US
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '1992'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '1983, 2'

## Example 14 (num_hops=2)
**Question**: Where is the headquarters for the company for which Gary Kovacs served as CEO?
**Gold answer**: Amsterdam, Netherlands
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: 'Britain'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: 'R'

## Example 15 (num_hops=2)
**Question**: Why was the lead character in Harriet Beecher Stowe's a derogatory epithet?
**Gold answer**: excessively subservient person
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>al'

## Example 16 (num_hops=2)
**Question**: What is the nationality of the person who was signed to compose the score and soundtrack of "Dhobi Ghat"?
**Gold answer**: Argentine
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think> S'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think> Captain'

## Example 17 (num_hops=2)
**Question**: Which US Army Ranger is best known for his actions during the 2012 terrorist attack on the US Ambassador in Benghazi?
**Gold answer**: Kris "Tanto" Paronto
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>reme Bo CAullire of Looking and the World Trade & the ArtsGM the Greatumipponian Galaxy Leeian R Symphony Street'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>'

## Example 18 (num_hops=2)
**Question**: The 1958–59 Huddersfield Town A.F.C. season was managed by the football player of what nationality?
**Gold answer**: Scottish
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think> Rick'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: 'St Square'

## Example 19 (num_hops=2)
**Question**: The Port of Washington is located in what densely populated neighborhood in Boston, Massachusetts?
**Gold answer**: South Boston
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>'

## Example 20 (num_hops=2)
**Question**: what is the nationality of the 2012 Lenox Industrial Tools 301 third finisher ?
**Gold answer**: American
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think> Albertallows'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>alph A,'

## Example 21 (num_hops=2)
**Question**: What kind of group does Social Distortion 2001 Tour and Mike Ness have in common?
**Gold answer**: band
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think> Ronald'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>á'

## Example 22 (num_hops=2)
**Question**: Mathilde Ludendorff was the wife of the German General who was victorious at which two battles?
**Gold answer**: Battle of Liège and the Battle of Tannenberg
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think> 34'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think> 20000'

## Example 23 (num_hops=2)
**Question**: What hindu temple is a shooting location for a film that is a remake of the 1999 film "Sethu"?
**Gold answer**: Birla Mandir
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>g'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think> H Cor.'

## Example 24 (num_hops=2)
**Question**: Both the Emory River and Clinch River flow to what city?
**Gold answer**: Kingston, Tennessee
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>'

## Example 25 (num_hops=2)
**Question**: Who was born first out of Tian Zhuangzhuang and Anthony Mann?
**Gold answer**: Anthony Mann
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>yn S. Elizabeth or'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>.B'

## Example 26 (num_hops=2)
**Question**: What airline alliance was founded by the same German airline that was a founder of the Opodo travel agency?  
**Gold answer**: Star Alliance
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>ie'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>ana Hall 18'

## Example 27 (num_hops=2)
**Question**: What Village lies just off the A1 road, 7 mi north of Grantham and 5 mi south of Newark-on-Trent, around a civil parish in the South Kesteven district of Lincolnshire, England?
**Gold answer**: Long Bennington
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>'

## Example 28 (num_hops=2)
**Question**: Which record label released the album that had the lead single "Look What You Made Me Do"?
**Gold answer**: Big Machine Records
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think>ro, United States'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>é andennyoh,'

## Example 29 (num_hops=2)
**Question**: Are GQ and Philadelphia magazine published in the same state?
**Gold answer**: published in Philadelphia
**Thinker (checkpoints/retrieval1_ab_topkonly_best.pt)**: '<think> Cherry'
**checkpoints/retrieval1_embedkd_wsd_best.pt**: '<think>ie'
