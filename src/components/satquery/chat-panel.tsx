'use client';

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';

import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';

import {
  Send,
  Sparkles,
  Trash2,
  AlertCircle,
  User,
  Bot,
  Loader2,
} from 'lucide-react';

import { useSatQueryStore } from '@/store/satquery';
import { useToast } from '@/hooks/use-toast';
import { shortId } from '@/lib/client-utils';

import type {
  AnalysisResult,
  ChatMessage,
} from '@/lib/types';

import { MarkdownLite } from './markdown-lite';


// =========================================================
// SUGGESTED QUERIES
// =========================================================

const SUGGESTED_QUERIES = [
  'Detect buildings in this image.',
  'Detect ships in this image.',
  'Find aircraft in this image.',
  'Detect roads in this image.',
  'Identify water bodies in this image.',
  'Describe the major geographic features.',
];


// =========================================================
// API TYPES
// =========================================================

interface AnalyzeResponse {
  analysis: AnalysisResult;

  rawAnswer: string;

  intent: string;

  intentLabel: string;

  modelUsed:
    | 'glm'
    | 'gemini';

  fallbackUsed: boolean;

  usedDetectionContext?: boolean;
}


interface DinoDetection {
  label: string;

  confidence: number;

  box: [
    number,
    number,
    number,
    number
  ];
}


interface DinoResponse {
  count: number;

  width: number;

  height: number;

  detections: DinoDetection[];
}


// =========================================================
// DECIDE WHEN TO USE GROUNDING DINO
// =========================================================

function shouldUseGroundingDino(
  query: string,
) {
  const q =
    query.toLowerCase();


  const detectionWords = [
    'detect',
    'find',
    'locate',
    'identify',
    'show',
    'mark',
    'count',
    'how many',
    'are there',
  ];


  const objectWords = [
    'building',
    'buildings',
    'urban',
    'structure',
    'structures',

    'road',
    'roads',
    'street',
    'streets',
    'highway',
    'highways',

    'car',
    'cars',

    'vehicle',
    'vehicles',
    'truck',
    'trucks',

    'ship',
    'ships',
    'vessel',
    'vessels',
    'boat',
    'boats',

    'aircraft',
    'airplane',
    'airplanes',
    'plane',
    'planes',
    'helicopter',

    'bridge',
    'bridges',

    'house',
    'houses',

    'water',
    'water body',
    'water bodies',
    'river',
    'lake',
  ];


  const hasDetectionWord =
    detectionWords.some(
      (word) =>
        q.includes(word),
    );


  const hasObjectWord =
    objectWords.some(
      (word) =>
        q.includes(word),
    );


  return (
    hasDetectionWord &&
    hasObjectWord
  );
}


// =========================================================
// BUILD GROUNDING DINO PROMPT
//
// Grounding DINO performs substantially better when given
// synonym-rich label chains separated by " . ".
// Each token group expands the model's text attention
// and reduces misclassification (e.g. ship → road).
// =========================================================

function buildDinoPrompt(
  query: string,
) {
  const q =
    query.toLowerCase();


  // Each entry is a synonym chain for one object class.
  const labelChains: string[] = [];


  // ---------------------------------------------------
  // Buildings / Structures
  // ---------------------------------------------------
  if (
    q.includes('building') ||
    q.includes('buildings') ||
    q.includes('urban') ||
    q.includes('structure') ||
    q.includes('structures')
  ) {
    labelChains.push(
      'building . structure . rooftop . ' +
      'warehouse . factory . residential building .',
    );
  }


  // ---------------------------------------------------
  // Roads / Streets
  // ---------------------------------------------------
  if (
    q.includes('road') ||
    q.includes('roads') ||
    q.includes('street') ||
    q.includes('streets') ||
    q.includes('highway') ||
    q.includes('highways')
  ) {
    labelChains.push(
      'road . street . highway . ' +
      'paved road . roadway . path . pathway .',
    );
  }


  // ---------------------------------------------------
  // Cars
  // ---------------------------------------------------
  if (
    q.includes('car') ||
    q.includes('cars')
  ) {
    labelChains.push(
      'car . automobile . sedan . hatchback .',
    );
  }


  // ---------------------------------------------------
  // Vehicles / Trucks
  // ---------------------------------------------------
  if (
    q.includes('vehicle') ||
    q.includes('vehicles') ||
    q.includes('truck') ||
    q.includes('trucks')
  ) {
    labelChains.push(
      'vehicle . car . truck . bus . van . lorry .',
    );
  }


  // ---------------------------------------------------
  // Ships / Vessels / Boats
  // ---------------------------------------------------
  if (
    q.includes('ship') ||
    q.includes('ships') ||
    q.includes('vessel') ||
    q.includes('vessels') ||
    q.includes('boat') ||
    q.includes('boats')
  ) {
    labelChains.push(
      'ship . vessel . cargo ship . tanker . ' +
      'boat . freighter . ferry .',
    );
  }


  // ---------------------------------------------------
  // Aircraft / Planes
  // ---------------------------------------------------
  if (
    q.includes('aircraft') ||
    q.includes('airplane') ||
    q.includes('airplanes') ||
    q.includes('plane') ||
    q.includes('planes') ||
    q.includes('helicopter')
  ) {
    labelChains.push(
      'aircraft . airplane . jet . plane . helicopter .',
    );
  }


  // ---------------------------------------------------
  // Bridges
  // ---------------------------------------------------
  if (
    q.includes('bridge') ||
    q.includes('bridges')
  ) {
    labelChains.push(
      'bridge . overpass . viaduct . flyover .',
    );
  }


  // ---------------------------------------------------
  // Houses
  // ---------------------------------------------------
  if (
    q.includes('house') ||
    q.includes('houses')
  ) {
    labelChains.push(
      'house . residential building . home . dwelling .',
    );
  }


  // ---------------------------------------------------
  // Water / Rivers / Lakes
  // ---------------------------------------------------
  if (
    q.includes('water') ||
    q.includes('river') ||
    q.includes('lake')
  ) {
    labelChains.push(
      'water . river . lake . reservoir . pond . waterway .',
    );
  }


  if (labelChains.length > 0) {
    // Join multiple classes with space separator.
    // Each chain already ends with a " ." token.
    return labelChains.join(' ');
  }


  return query;
}


// =========================================================
// DETECTION COLORS
// =========================================================

function getDetectionColor(
  label: string,
) {
  const value =
    label.toLowerCase();


  if (
    value.includes(
      'building',
    )
  ) {
    return '#f97316';
  }


  if (
    value.includes(
      'road',
    )
  ) {
    return '#a8a29e';
  }


  if (
    value.includes(
      'water',
    ) ||
    value.includes(
      'river',
    ) ||
    value.includes(
      'lake',
    )
  ) {
    return '#06b6d4';
  }


  if (
    value.includes(
      'vegetation',
    ) ||
    value.includes(
      'forest',
    )
  ) {
    return '#22c55e';
  }


  if (
    value.includes(
      'car',
    ) ||
    value.includes(
      'vehicle',
    )
  ) {
    return '#eab308';
  }


  if (
    value.includes(
      'ship',
    ) ||
    value.includes(
      'vessel',
    )
  ) {
    return '#8b5cf6';
  }


  if (
    value.includes(
      'aircraft',
    ) ||
    value.includes(
      'plane',
    )
  ) {
    return '#ec4899';
  }


  if (
    value.includes(
      'bridge',
    )
  ) {
    return '#3b82f6';
  }


  return '#ef4444';
}


// =========================================================
// DETECTION INTENT
// =========================================================

function getDetectionIntent(
  query: string,
): AnalysisResult['intent'] {

  const q =
    query.toLowerCase();


  if (
    q.includes(
      'building',
    ) ||
    q.includes(
      'buildings',
    )
  ) {
    return 'building_detection';
  }


  if (
    q.includes(
      'road',
    ) ||
    q.includes(
      'roads',
    )
  ) {
    return 'road_detection';
  }


  if (
    q.includes(
      'ship',
    ) ||
    q.includes(
      'ships',
    ) ||
    q.includes(
      'vessel',
    ) ||
    q.includes(
      'vessels',
    )
  ) {
    return 'ship_detection';
  }


  if (
    q.includes(
      'aircraft',
    ) ||
    q.includes(
      'airplane',
    ) ||
    q.includes(
      'airplanes',
    ) ||
    q.includes(
      'plane',
    ) ||
    q.includes(
      'planes',
    )
  ) {
    return 'aircraft_detection';
  }


  if (
    q.includes(
      'vehicle',
    ) ||
    q.includes(
      'vehicles',
    ) ||
    q.includes(
      'car',
    ) ||
    q.includes(
      'cars',
    )
  ) {
    return 'vehicle_detection';
  }


  if (
    q.includes(
      'bridge',
    ) ||
    q.includes(
      'bridges',
    )
  ) {
    return 'bridge_detection';
  }


  if (
    q.includes(
      'water',
    ) ||
    q.includes(
      'river',
    ) ||
    q.includes(
      'lake',
    )
  ) {
    return 'water_detection';
  }


  return 'image_understanding';
}


// =========================================================
// CONVERT DINO RESPONSE TO ANALYSIS
// =========================================================

function convertDinoToAnalysis(
  data: DinoResponse,
  query: string,
): AnalysisResult {

  // -------------------------------------------------------
  // Bounding boxes
  // -------------------------------------------------------

  const regions =
    data.detections.map(
      (detection) => {

        const [
          x1,
          y1,
          x2,
          y2,
        ] =
          detection.box;


        return {
          label:
            detection.label,

          color:
            getDetectionColor(
              detection.label,
            ),

          rect: [
            x1 /
              data.width,

            y1 /
              data.height,

            (x2 - x1) /
              data.width,

            (y2 - y1) /
              data.height,
          ] as [
            number,
            number,
            number,
            number,
          ],

          confidence:
            detection.confidence,
        };
      },
    );


  // -------------------------------------------------------
  // Group detections
  // -------------------------------------------------------

  const grouped =
    new Map<
      string,
      {
        count: number;

        confidences:
          number[];
      }
    >();


  for (
    const detection
    of data.detections
  ) {

    const label =
      detection.label
        .toLowerCase()
        .trim();


    const current =
      grouped.get(
        label,
      );


    if (current) {

      current.count +=
        1;


      current.confidences.push(
        detection.confidence,
      );

    } else {

      grouped.set(
        label,
        {
          count:
            1,

          confidences: [
            detection.confidence,
          ],
        },
      );
    }
  }


  // -------------------------------------------------------
  // Object summary
  // -------------------------------------------------------

  const objectsDetected =
    Array.from(
      grouped.entries(),
    ).map(
      (
        [
          label,
          info,
        ],
      ) => {

        const averageConfidence =
          info.confidences.reduce(
            (
              sum,
              value,
            ) =>
              sum +
              value,
            0,
          ) /
          info.confidences.length;


        return {
          class:
            label,

          count:
            info.count,

          confidence:
            averageConfidence,
        };
      },
    );


  // -------------------------------------------------------
  // Overall detector confidence
  // -------------------------------------------------------

  const overallConfidence =
    data.detections.length >
    0

      ? data.detections.reduce(
          (
            sum,
            detection,
          ) =>
            sum +
            detection.confidence,
          0,
        ) /
        data.detections.length

      : 0;


  // -------------------------------------------------------
  // Temporary DINO answer
  //
  // This will later be replaced by the
  // GLM/Gemini human-readable explanation.
  // -------------------------------------------------------

  let answer =
    '';


  if (
    data.count ===
    0
  ) {

    answer =
      'Grounding DINO did not detect any matching objects in this image.';

  } else {

    const summary =
      objectsDetected
        .map(
          (object) =>
            `${object.class} ×${object.count}`,
        )
        .join(
          ', ',
        );


    answer =
      `Grounding DINO detected **${data.count} object${
        data.count ===
        1
          ? ''
          : 's'
      }** in the image.\n\n` +

      `Detected: ${summary}.\n\n` +

      'The detected regions are marked on the image.';
  }


  return {
    answer,

    intent:
      getDetectionIntent(
        query,
      ),

    objectsDetected,

    confidence:
      overallConfidence,

    coverage:
      [],

    regions,
  };
}


// =========================================================
// MAIN CHAT PANEL
// =========================================================

export function ChatPanel() {

  const activeImage =
    useSatQueryStore(
      (s) =>
        s.activeImage,
    );


  const messages =
    useSatQueryStore(
      (s) =>
        s.messages,
    );


  const isAnalyzing =
    useSatQueryStore(
      (s) =>
        s.isAnalyzing,
    );


  const addMessage =
    useSatQueryStore(
      (s) =>
        s.addMessage,
    );


  const updateMessage =
    useSatQueryStore(
      (s) =>
        s.updateMessage,
    );


  const setIsAnalyzing =
    useSatQueryStore(
      (s) =>
        s.setIsAnalyzing,
    );


  const setLatestAnalysis =
    useSatQueryStore(
      (s) =>
        s.setLatestAnalysis,
    );


  const clearChat =
    useSatQueryStore(
      (s) =>
        s.clearChat,
    );


  const { toast } =
    useToast();


  const [
    input,
    setInput,
  ] =
    useState('');


  const [
    selectedModel,
    setSelectedModel,
  ] =
    useState<
      'glm' |
      'gemini'
    >('glm');


  /*
   * Used only for displaying which
   * pipeline answered the request.
   */

  const [
    lastModelUsed,
    setLastModelUsed,
  ] =
    useState<
      | 'glm'
      | 'gemini'
      | 'dino-glm'
      | 'dino-gemini'
      | null
    >(null);


  const [
    lastFallbackUsed,
    setLastFallbackUsed,
  ] =
    useState(
      false,
    );


  const scrollRef =
    useRef<HTMLDivElement>(
      null,
    );


  const textareaRef =
    useRef<HTMLTextAreaElement>(
      null,
    );


  // =======================================================
  // AUTO SCROLL
  // =======================================================

  useEffect(
    () => {

      if (
        scrollRef.current
      ) {

        scrollRef.current.scrollTop =
          scrollRef.current.scrollHeight;
      }

    },
    [
      messages,
      isAnalyzing,
    ],
  );


  // =======================================================
  // SUBMIT QUERY
  // =======================================================

  const submitQuery =
    useCallback(

      async (
        text: string,
      ) => {

        const trimmed =
          text.trim();


        if (!trimmed) {
          return;
        }


        if (!activeImage) {

          toast({
            variant:
              'destructive',

            title:
              'No image selected',

            description:
              'Please upload or select a satellite image first.',
          });

          return;
        }


        if (
          isAnalyzing
        ) {
          return;
        }


        // ---------------------------------------------------
        // Add user message
        // ---------------------------------------------------

        const userMsg:
          ChatMessage = {

          id:
            shortId(
              'u-',
            ),

          role:
            'user',

          content:
            trimmed,

          createdAt:
            new Date()
              .toISOString(),
        };


        addMessage(
          userMsg,
        );


        setInput(
          '',
        );


        setIsAnalyzing(
          true,
        );


        // ---------------------------------------------------
        // Add pending assistant message
        // ---------------------------------------------------

        const assistantMsgId =
          shortId(
            'a-',
          );


        addMessage({
          id:
            assistantMsgId,

          role:
            'assistant',

          content:
            '',

          createdAt:
            new Date()
              .toISOString(),

          pending:
            true,
        });


        try {

          // =================================================
          // BUILD HISTORY ONCE
          // =================================================

          const history =
            messages
              .filter(
                (m) =>
                  !m.pending &&
                  !m.error,
              )
              .slice(
                -6,
              )
              .map(
                (m) => ({
                  role:
                    m.role,

                  content:
                    m.content,
                }),
              );


          // =================================================
          // GROUNDING DINO PIPELINE
          // =================================================

          if (
            shouldUseGroundingDino(
              trimmed,
            )
          ) {

            // -----------------------------------------------
            // Step A:
            // Convert question into detector prompt
            // -----------------------------------------------

            const dinoPrompt =
              buildDinoPrompt(
                trimmed,
              );


            console.log(
              '[SatQuery] Step 1: Grounding DINO',
            );


            console.log(
              '[SatQuery] DINO prompt:',
              dinoPrompt,
            );


            // -----------------------------------------------
            // Step B:
            // Call Grounding DINO
            // -----------------------------------------------

            const dinoResponse =
              await fetch(
                '/api/detect',
                {
                  method:
                    'POST',

                  headers: {
                    'Content-Type':
                      'application/json',
                  },

                  body:
                    JSON.stringify({
                      imageDataUrl:
                        activeImage.dataUrl,

                      prompt:
                        dinoPrompt,
                    }),
                },
              );


            if (
              !dinoResponse.ok
            ) {

              let errorMessage =
                `Grounding DINO error: HTTP ${dinoResponse.status}`;


              try {

                const errorData =
                  await dinoResponse.json();


                if (
                  errorData.error
                ) {

                  errorMessage =
                    errorData.error;
                }

              } catch {
                // Ignore invalid error JSON
              }


              throw new Error(
                errorMessage,
              );
            }


            // -----------------------------------------------
            // Step C:
            // Read DINO structured detections
            // -----------------------------------------------

            const dinoData =
              (
                await dinoResponse.json()
              ) as DinoResponse;


            console.log(
              '[SatQuery] DINO detections:',
              dinoData,
            );


            // -----------------------------------------------
            // Step D:
            // Convert boxes into SatQuery UI format
            // -----------------------------------------------

            const dinoAnalysis =
              convertDinoToAnalysis(
                dinoData,
                trimmed,
              );


            /*
             * IMPORTANT:
             *
             * At this point we already have:
             *
             * - exact detector count
             * - detector labels
             * - detector confidence
             * - detector bounding boxes
             *
             * Now we pass those detections
             * to GLM/Gemini.
             */


            console.log(
              `[SatQuery] Step 2: Sending DINO results to ${selectedModel}`,
            );


            // -----------------------------------------------
            // Step E:
            // DINO -> /api/analyze -> GLM/Gemini
            // -----------------------------------------------

            const reasoningResponse =
              await fetch(
                '/api/analyze',
                {
                  method:
                    'POST',

                  headers: {
                    'Content-Type':
                      'application/json',
                  },

                  body:
                    JSON.stringify({
                      imageDataUrl:
                        activeImage.dataUrl,

                      secondImageDataUrl:
                        activeImage.secondDataUrl,

                      query:
                        trimmed,

                      history,

                      model:
                        selectedModel,

                      /*
                       * THIS IS THE IMPORTANT NEW PART.
                       */

                      detectionContext: {
                        count:
                          dinoData.count,

                        width:
                          dinoData.width,

                        height:
                          dinoData.height,

                        detections:
                          dinoData.detections,
                      },
                    }),
                },
              );


            // -----------------------------------------------
            // If VLM reasoning fails
            // -----------------------------------------------

            if (
              !reasoningResponse.ok
            ) {

              let errorMessage =
                `AI reasoning error: HTTP ${reasoningResponse.status}`;


              try {

                const errorData =
                  await reasoningResponse.json();


                if (
                  errorData.error
                ) {

                  errorMessage =
                    errorData.error;
                }

              } catch {
                // Ignore invalid JSON
              }


              /*
               * IMPORTANT FALLBACK:
               *
               * DINO itself worked.
               *
               * So don't throw away the detection result
               * just because the VLM explanation failed.
               */

              console.warn(
                '[SatQuery] VLM explanation failed. Showing DINO-only result.',
                errorMessage,
              );


              updateMessage(
                assistantMsgId,
                {
                  content:
                    dinoAnalysis.answer,

                  analysis:
                    dinoAnalysis,

                  pending:
                    false,
                },
              );


              setLatestAnalysis(
                dinoAnalysis,
              );


              /*
               * We still show that only DINO
               * completed successfully.
               */

              setLastModelUsed(
                selectedModel ===
                'glm'

                  ? 'dino-glm'

                  : 'dino-gemini',
              );


              setLastFallbackUsed(
                false,
              );


              toast({
                title:
                  'Detection completed',

                description:
                  'Grounding DINO worked, but the AI explanation was unavailable.',
              });


              return;
            }


            // -----------------------------------------------
            // Step F:
            // Read GLM/Gemini explanation
            // -----------------------------------------------

            const reasoningData =
              (
                await reasoningResponse.json()
              ) as AnalyzeResponse;


            console.log(
              '[SatQuery] Reasoning result:',
              reasoningData,
            );


            /*
             * VERY IMPORTANT:
             *
             * We DO NOT use the VLM's object counts,
             * boxes, or detector confidence.
             *
             * Grounding DINO remains source of truth
             * for object detection.
             *
             * The VLM only gives the explanation.
             */


            // -----------------------------------------------
            // Step G:
            // Merge DINO + VLM
            // -----------------------------------------------

            const finalAnalysis:
              AnalysisResult = {

              /*
               * Preserve any useful extra fields
               * produced by parseAnalysis.
               */

              ...reasoningData.analysis,


              /*
               * HUMAN-READABLE EXPLANATION
               *
               * comes from GLM/Gemini.
               */

              answer:
                reasoningData.analysis
                  .answer ||
                reasoningData.rawAnswer ||
                dinoAnalysis.answer,


              /*
               * DETECTION INTENT
               *
               * Keep our reliable intent mapping.
               */

              intent:
                dinoAnalysis.intent,


              /*
               * OBJECT COUNTS
               *
               * Grounding DINO only.
               */

              objectsDetected:
                dinoAnalysis.objectsDetected,


              /*
               * DETECTION CONFIDENCE
               *
               * Grounding DINO only.
               */

              confidence:
                dinoAnalysis.confidence,


              /*
               * BOUNDING BOXES
               *
               * Grounding DINO only.
               */

              regions:
                dinoAnalysis.regions,


              /*
               * Keep coverage from VLM if it
               * produced something useful.
               *
               * Otherwise default to DINO's [].
               */

              coverage:
                reasoningData.analysis
                  .coverage ??
                dinoAnalysis.coverage,
            };


            // -----------------------------------------------
            // Step H:
            // Display final combined result
            // -----------------------------------------------

            updateMessage(
              assistantMsgId,
              {
                content:
                  finalAnalysis.answer,

                analysis:
                  finalAnalysis,

                pending:
                  false,
              },
            );


            setLatestAnalysis(
              finalAnalysis,
            );


            // -----------------------------------------------
            // Show actual pipeline used
            // -----------------------------------------------

            if (
              reasoningData.modelUsed ===
              'glm'
            ) {

              setLastModelUsed(
                'dino-glm',
              );

            } else {

              setLastModelUsed(
                'dino-gemini',
              );
            }


            setLastFallbackUsed(
              reasoningData.fallbackUsed,
            );


            console.log(
              '[SatQuery] Complete pipeline:',
              `Grounding DINO -> ${reasoningData.modelUsed}`,
            );


            return;
          }


          // =================================================
          // NORMAL GLM / GEMINI PIPELINE
          // =================================================

          console.log(
            `[SatQuery] Direct ${selectedModel} analysis`,
          );


          const res =
            await fetch(
              '/api/analyze',
              {
                method:
                  'POST',

                headers: {
                  'Content-Type':
                    'application/json',
                },

                body:
                  JSON.stringify({
                    imageDataUrl:
                      activeImage.dataUrl,

                    secondImageDataUrl:
                      activeImage.secondDataUrl,

                    query:
                      trimmed,

                    history,

                    model:
                      selectedModel,
                  }),
              },
            );


          if (
            !res.ok
          ) {

            let errText =
              `HTTP ${res.status}`;


            try {

              const errJson =
                (
                  await res.json()
                ) as {
                  error?: string;
                };


              if (
                errJson.error
              ) {

                errText =
                  errJson.error;
              }

            } catch {
              // Ignore parsing error
            }


            throw new Error(
              errText,
            );
          }


          const data =
            (
              await res.json()
            ) as AnalyzeResponse;


          // -------------------------------------------------
          // Model label
          // -------------------------------------------------

          setLastModelUsed(
            data.modelUsed,
          );


          setLastFallbackUsed(
            data.fallbackUsed,
          );


          // -------------------------------------------------
          // Show response
          // -------------------------------------------------

          updateMessage(
            assistantMsgId,
            {
              content:
                data.analysis.answer,

              analysis:
                data.analysis,

              pending:
                false,
            },
          );


          setLatestAnalysis(
            data.analysis,
          );


        } catch (err) {

          const msg =
            err instanceof Error
              ? err.message
              : 'Failed to analyze';


          updateMessage(
            assistantMsgId,
            {
              content:
                '',

              pending:
                false,

              error:
                msg,
            },
          );


          toast({
            variant:
              'destructive',

            title:
              'Analysis failed',

            description:
              msg,
          });


        } finally {

          setIsAnalyzing(
            false,
          );
        }

      },

      [
        activeImage,
        addMessage,
        updateMessage,
        setIsAnalyzing,
        setLatestAnalysis,
        messages,
        isAnalyzing,
        toast,
        selectedModel,
      ],
    );


  // =======================================================
  // KEYBOARD SHORTCUT
  // =======================================================

  const onKeyDown = (
    e:
      React.KeyboardEvent<
        HTMLTextAreaElement
      >,
  ) => {

    if (
      (
        e.metaKey ||
        e.ctrlKey
      ) &&
      e.key ===
        'Enter'
    ) {

      e.preventDefault();


      submitQuery(
        input,
      );
    }
  };


  const showEmpty =
    messages.length ===
    0;


  // =======================================================
  // UI
  // =======================================================

  return (

    <div className="flex h-full min-h-[500px] flex-col">

      {/* ===================================================
          HEADER
          =================================================== */}

      <div className="flex items-center justify-between gap-2 border-b px-3 py-2.5">

        <div className="flex min-w-0 items-center gap-2">

          <div className="flex size-7 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">

            <Sparkles className="size-4" />

          </div>


          <div className="min-w-0 leading-tight">

            <p className="text-sm font-semibold">
              AI Assistant
            </p>


            <p className="text-[11px] text-muted-foreground">
              Ask questions about the image
            </p>

          </div>

        </div>


        <div className="flex shrink-0 items-center gap-2">

          {/* ===============================================
              MODEL SELECTOR
              =============================================== */}

          <select

            value={
              selectedModel
            }

            onChange={(
              e,
            ) => {

              setSelectedModel(
                e.target.value as
                  | 'glm'
                  | 'gemini',
              );


              setLastModelUsed(
                null,
              );


              setLastFallbackUsed(
                false,
              );
            }}

            disabled={
              isAnalyzing
            }

            aria-label="Select AI model"

            className="h-8 rounded-md border bg-background px-2 text-xs font-medium outline-none transition-colors focus:ring-2 focus:ring-primary/30 disabled:cursor-not-allowed disabled:opacity-50"
          >

            <option value="glm">
              GLM 4.6 Flash
            </option>


            <option value="gemini">
              Gemini
            </option>

          </select>


          {/* ===============================================
              CLEAR
              =============================================== */}

          {messages.length >
            0 && (

            <Button

              size="sm"

              variant="ghost"

              onClick={() => {

                clearChat();


                setLastModelUsed(
                  null,
                );


                setLastFallbackUsed(
                  false,
                );


                toast({
                  title:
                    'Conversation cleared',
                });

              }}

              className="gap-1.5 text-muted-foreground hover:text-foreground"
            >

              <Trash2 className="size-3.5" />

              Clear

            </Button>

          )}

        </div>

      </div>


      {/* ===================================================
          MESSAGES
          =================================================== */}

      <div

        ref={
          scrollRef
        }

        className="satquery-scroll flex-1 space-y-4 overflow-y-auto p-3"
      >

        {showEmpty && (

          <EmptyState

            hasImage={
              Boolean(
                activeImage,
              )
            }

            onPick={(
              q,
            ) =>
              submitQuery(
                q,
              )
            }
          />

        )}


        {messages.map(
          (m) => (

            <MessageBubble

              key={
                m.id
              }

              message={
                m
              }
            />

          ),
        )}


        {isAnalyzing && (

          <div className="flex items-center gap-2 pl-9 text-xs text-muted-foreground">

            <Loader2 className="size-3 animate-spin" />

            SatQuery AI is analyzing the satellite image…

          </div>

        )}

      </div>


      {/* ===================================================
          INPUT
          =================================================== */}

      <div className="border-t p-3">

        <div className="relative">

          <Textarea

            ref={
              textareaRef
            }

            value={
              input
            }

            onChange={(
              e,
            ) =>
              setInput(
                e.target.value,
              )
            }

            onKeyDown={
              onKeyDown
            }

            placeholder={
              activeImage

                ? 'Ask anything about this satellite image…  (⌘/Ctrl+↵ to send)'

                : 'Select an image first to start asking questions…'
            }

            disabled={
              isAnalyzing ||
              !activeImage
            }

            rows={
              2
            }

            className="resize-none pr-12 text-sm"
          />


          <Button

            type="button"

            size="icon"

            className="absolute bottom-2 right-2 size-8 rounded-md"

            onClick={() =>
              submitQuery(
                input,
              )
            }

            disabled={
              isAnalyzing ||
              !input.trim() ||
              !activeImage
            }

            aria-label="Send query"
          >

            <Send className="size-4" />

          </Button>

        </div>


        {/* ===============================================
            PIPELINE STATUS
            =============================================== */}

        <p className="mt-1.5 text-[10px] text-muted-foreground">

          Press{' '}

          <kbd className="rounded bg-muted px-1 py-0.5 font-mono text-[9px]">

            ⌘/Ctrl + ↵

          </kbd>{' '}

          to send · Selected:{' '}


          {selectedModel ===
          'glm'

            ? 'GLM 4.6 Flash'

            : 'Gemini'}


          {lastModelUsed && (

            <>

              {' · '}Used:{' '}


              {lastModelUsed ===
              'glm'

                ? 'GLM 4.6 Flash'

                : lastModelUsed ===
                    'gemini'

                  ? 'Gemini'

                  : lastModelUsed ===
                      'dino-glm'

                    ? 'Grounding DINO → GLM 4.6 Flash'

                    : 'Grounding DINO → Gemini'}

            </>

          )}


          {lastFallbackUsed &&
            ' · Fallback'}

        </p>

      </div>

    </div>
  );
}


// =========================================================
// EMPTY STATE
// =========================================================

function EmptyState({
  hasImage,
  onPick,
}: {
  hasImage:
    boolean;

  onPick: (
    q: string,
  ) => void;
}) {

  return (

    <div className="flex flex-col items-center gap-4 py-6 text-center">

      <div className="flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary">

        <Bot className="size-6" />

      </div>


      <div className="space-y-1 px-4">

        <p className="text-sm font-semibold">

          {hasImage

            ? 'Ready to analyze your image'

            : 'Welcome to SatQuery AI'}

        </p>


        <p className="text-xs text-muted-foreground">

          {hasImage

            ? 'Try one of the suggested queries below to get started.'

            : 'Upload a satellite image, then ask any question in natural language.'}

        </p>

      </div>


      {hasImage && (

        <div className="grid w-full max-w-sm gap-1.5">

          {SUGGESTED_QUERIES.map(
            (q) => (

              <button

                key={
                  q
                }

                type="button"

                onClick={() =>
                  onPick(
                    q,
                  )
                }

                className="group flex items-center gap-2 rounded-md border bg-card/50 px-2.5 py-1.5 text-left text-xs transition-all hover:border-primary/50 hover:bg-accent/50"
              >

                <Sparkles className="size-3 shrink-0 text-primary/70 group-hover:text-primary" />

                <span>
                  {q}
                </span>

              </button>

            ),
          )}

        </div>

      )}

    </div>
  );
}


// =========================================================
// MESSAGE BUBBLE
// =========================================================

function MessageBubble({
  message,
}: {
  message:
    ChatMessage;
}) {

  const isUser =
    message.role ===
    'user';


  return (

    <div

      className={cn(
        'flex w-full gap-2 animate-fade-in-up',

        isUser
          ? 'flex-row-reverse'
          : 'flex-row',
      )}
    >

      <div

        className={cn(
          'flex size-7 shrink-0 items-center justify-center rounded-full',

          isUser

            ? 'bg-secondary text-secondary-foreground'

            : 'bg-primary/10 text-primary',
        )}
      >

        {isUser ? (

          <User className="size-3.5" />

        ) : (

          <Bot className="size-3.5" />

        )}

      </div>


      <div

        className={cn(
          'max-w-[85%] space-y-2 rounded-lg px-3 py-2 text-sm',

          isUser

            ? 'bg-primary text-primary-foreground'

            : 'bg-card border',
        )}
      >

        {message.pending ? (

          <div className="flex items-center gap-2 text-muted-foreground">

            <Loader2 className="size-3 animate-spin" />

            <span className="text-xs italic">
              Analyzing image…
            </span>

          </div>

        ) : message.error ? (

          <div className="flex items-start gap-2 text-destructive">

            <AlertCircle className="size-4 shrink-0 translate-y-0.5" />


            <div className="space-y-1">

              <p className="text-xs font-semibold">
                Analysis failed
              </p>


              <p className="text-xs">
                {message.error}
              </p>

            </div>

          </div>

        ) : (

          <>

            {/* Human-readable answer */}

            <MarkdownLite
              text={
                message.content
              }
            />


            {/* Detected object badges */}

            {message.analysis &&
              message.analysis
                .objectsDetected
                .length > 0 && (

                <ObjectsDetectedBadges
                  analysis={
                    message.analysis
                  }
                />

              )}


            {/* Intent + confidence */}

            {message.analysis && (

              <div className="flex flex-wrap items-center gap-2 pt-1 text-[10px] text-muted-foreground">

                <span className="rounded bg-secondary/60 px-1.5 py-0.5 font-medium">

                  Intent:{' '}

                  {message.analysis.intent.replace(
                    /_/g,
                    ' ',
                  )}

                </span>


                <span className="rounded bg-secondary/60 px-1.5 py-0.5 font-medium">

                  Detection Confidence:{' '}

                  {Math.round(
                    message.analysis
                      .confidence *
                      100,
                  )}

                  %

                </span>

              </div>

            )}

          </>

        )}

      </div>

    </div>
  );
}


// =========================================================
// DETECTED OBJECT BADGES
// =========================================================

function ObjectsDetectedBadges({
  analysis,
}: {
  analysis:
    AnalysisResult;
}) {

  return (

    <div className="flex flex-wrap gap-1 pt-1">

      {analysis.objectsDetected.map(
        (
          object,
          index,
        ) => (

          <span

            key={
              `${object.class}-${index}`
            }

            className="inline-flex items-center gap-1 rounded-full bg-secondary/70 px-2 py-0.5 text-[10px] font-medium"
          >

            <span className="capitalize">
              {object.class}
            </span>


            {typeof object.count ===
              'number' && (

              <span className="opacity-70">
                ×{object.count}
              </span>

            )}


            <span className="font-semibold text-primary">

              {Math.round(
                object.confidence *
                  100,
              )}

              %

            </span>

          </span>

        ),
      )}

    </div>
  );
}