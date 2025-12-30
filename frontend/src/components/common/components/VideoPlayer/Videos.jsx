import { useState, useEffect, useRef, useContext } from "react";
import { Box, Button, Card, Fab, FormControl, InputLabel, Menu, MenuItem, Select, Slider, Typography, Accordion, AccordionSummary, AccordionDetails, Paper } from "@mui/material";
import { getChainImagesKey, getImage, getImageKey, getImageKey_, imagePath } from "@/utils/imagePath";
import clsx from "clsx";
import Selecto from "react-selecto";
import { useStore } from "@/stores/questions";
import { useStoreImages } from "@/stores/blobs";
import { get } from "idb-keyval";
import useLocalStorageState from "use-local-storage-state";
import { Disclosure, DisclosureButton, DisclosurePanel } from '@headlessui/react'

export default function Videos({ handleOpen }) {
  // const [sortOption, setSortOption] = useLocalStorageState("sortOption", {defaultValue: "g"}); // d: default g: groupvid hc: high score
  const defaultCols = "grid-cols-5";
  const zoomedOutCols = "grid-cols-7";
  const [cols, setCols] = useState(defaultCols);

  useEffect(() => {
    const update = () => {
      if (window.devicePixelRatio <= 0.5) {
        setCols(zoomedOutCols);
      } else {
        setCols(defaultCols);
      }
    }
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  })

  // const [groupOption, setGroupOption] = useLocalStorageState("groupOption", {defaultValue: "n"}); // n: nogroup g: group hc: high score
  const { getCurrentQuestion, setCurrentQuestion, updateQuestionField, questions, currentQuestionId, undo, redo, sortOption, setSortOption, groupOption, setGroupOption, searchQuestions } = useStore();
  const currentQuestion = getCurrentQuestion();
  const images = currentQuestion.images;
  const searchImages = searchQuestions;

  // if searchImages.items exists then multiple
  const searchMode = searchImages[0]?.items ? "multiple" : "single";
  let searchImagesChain = searchQuestions;
  let groupedSearchImagesChain = [];

  if (searchMode === "multiple") {
    const map = new Map()
    searchImagesChain.forEach(imggroup => {
      let group;
      if (groupOption === "g") {
        group = imggroup.items[0].group_id;
      } else if (groupOption === "hc") {
        group = imggroup.score > 0.95 ? "Very High" : imggroup.score > 0.9 ? "High" : imggroup.score > 0.8 ? "Good" : imggroup.score > 0.7 ? "Average" : "Low";
      }
      if (!map.has(group)) {
        map.set(group, []);
      }
      map.get(group).push(imggroup.items);
    })


    // console.log(map)
    const groupSearchChainImagesSortFunction = (a, b) => {
      // console.log(a, b)
      const aGroup = a.group;
      const bGroup = b.group;
      if (groupOption === "g") {
        return aGroup < bGroup;
      } else if (groupOption === "hc") {
        return -rank[aGroup] + rank[bGroup];
      } else
        return 0;
    }

    const sortChainImagesFunction = (a, b) => {
      if (sortOption === "g") {
        return a.group_id - b.group_id || a.video_id - b.video_id || a.keyframe_id - b.keyframe_id;
      } else if (sortOption === "hc") {
        return b.score - a.score;
      } else
        return 0;

    }


    groupedSearchImagesChain = Array.from(map.entries()).map(([group, images]) => ({ group, images: images.sort(sortChainImagesFunction) })).sort(groupSearchChainImagesSortFunction);
  }

  const map = new Map()

  searchImages.forEach(img => {
    let group;
    if (groupOption === "g") {
      group = img.group_id;
    } else if (groupOption === "hc") {
      group = img.score > 0.95 ? "Very High" : img.score > 0.9 ? "High" : img.score > 0.8 ? "Good" : img.score > 0.7 ? "Average" : "Low";
    }
    if (!map.has(group)) {
      map.set(group, []);
    }
    map.get(group).push(img);
  })

  const groupSearchImagesSortFunction = (a, b) => {
    const aGroup = a.group;
    const bGroup = b.group;
    if (groupOption === "g") {
      return aGroup < bGroup;
    } else if (groupOption === "hc") {
      return -rank[aGroup] + rank[bGroup];
    } else
      return 0;
  }

  const sortImagesFunction = (a, b) => {
    if (sortOption === "g") {
      return a.group_id - b.group_id || a.video_id - b.video_id || a.keyframe_id - b.keyframe_id;
    } else if (sortOption === "hc") {
      return b.score - a.score;
    } else
      return 0;

  }


  let groupedSearchImages = Array.from(map.entries()).map(([group, images]) => ({ group, images: images.sort(sortImagesFunction) })).sort(groupSearchImagesSortFunction);

  const undoArray = currentQuestion.undoSearchArray;
  const redoArray = currentQuestion.redoSearchArray;

  const ref = useRef(null);

  const { blobs, setBlobs } = useStoreImages();

  let getKey = getImageKey_;
  if (searchMode == "multiple") getKey = getChainImagesKey;

  // const [blobUrls, setBlobUrls] = useState({});
  useEffect(() => {
    let isMounted = true;

    async function loadBlobs() {
      const urls = {};
      const uniqueMap = new Map();
      [...searchImages, ...images].forEach(img => {
        const imageKey = getImageKey(img.keyframe_id, img.video_id, img.group_id);
        if (!uniqueMap.has(imageKey)) uniqueMap.set(imageKey, img);
      });
      const allImages = [...uniqueMap.values()];
      for (const img of allImages) {
        if (img.blobKey) {
          const val = await get(img.blobKey);
          if (val) {
            urls[img.blobKey] = URL.createObjectURL(val);
          }
        }
      }
      setBlobs(urls);
    }
    loadBlobs();

    return () => {
      isMounted = false;
      Object.values(blobs).forEach(URL.revokeObjectURL);
    };
  }, [images]);


  const selectoRef = useRef(null);


  useEffect(() => {
    const handleKeyDown = (e) => {
      const isFocusedInside = ref.current && (ref.current === document.activeElement || ref.current.contains(document.activeElement));
      if (!isFocusedInside) return;

      // if (e.keyCode == 46 || e.keyCode == 8 || e.key.toLowerCase() == "d") {
      //   const selectedElements = document.querySelectorAll("#selecto .image.selected");
      //   if (!selectedElements.length) return;
      //   // delete selected elements by sorting them by key
      //   const selectedKeys = Array.from(selectedElements).map((el) => {
      //     return el.getAttribute("data-key");
      //   });
      //   const newSortedImages = searchImages.filter(image => !selectedKeys.includes(`${image.keyframe_id}-${image.video_id}-${image.group_id}`));
      //   updateQuestionField({
      //     'searchImages': newSortedImages,
      //   });
      // }

      // control + a, select all
      if (e.ctrlKey && e.key.toLowerCase() === "a") {
        e.preventDefault();
        const allElements = document.querySelectorAll("#selecto .image.search");
        // give all elements selected class
        allElements.forEach(el => {
          el.classList.add("selected");
        });
        selectoRef.current.setSelectedTargets(Array.from(allElements));
      }

      // if (e.ctrlKey && e.key.toLowerCase() === "z") {
      //   e.preventDefault();
      //   undo();
      // }
      // if (e.ctrlKey && e.key.toLowerCase() === "y") {
      //   e.preventDefault();
      //   redo();
      // }
    }
    document.addEventListener("keydown", handleKeyDown);

    // const handleDragDown = (e) => {
    //     if (e.keyCode == 2){
    //         e.preventDefault();
    //         const selectedElements = document.querySelectorAll("#selecto .image.selected");

    //     }
    // }

    // document.addEventListener("keydown", handleDragDown);
    const handleDragUp = (e) => {
      // if (ref.current && !ref.current.contains(e.target)) return;
      if (e.button !== 2) return; // Right-click only

      // hình chọn 
      const selectedElements = document.querySelectorAll("#selecto .image.selected");
      if (!selectedElements.length) return;
      // container của hình chọn
      const sourceContainer = selectedElements[0].getAttribute("data-container");

      // vị trí paste tới
      const currentElementMouseOn = document.elementFromPoint(e.clientX, e.clientY);
      // check xem có ở ngoài cửa sổ không
      if (!currentElementMouseOn) return;

      // check xem chuột có đang nằm trong 1 trong 2 container không
      const insideSearchImages = e.target.closest(".searchImages");
      const insideImages = e.target.closest(".images");
      const toContainer = insideSearchImages ? "searchImages" : insideImages ? "images" : null;

      if (!toContainer) return;


      // check xem 2 container giống hay khác

      const mode = sourceContainer === toContainer ? "same" : "different";
      // lấy data 2 container

      const sourceContainerData = sourceContainer === "images" ? images : searchImages;
      const toContainerData = toContainer === "images" ? images : searchImages;

      // data đã chọn 
      const selectedKeys = Array.from(selectedElements).map(el =>
        el.getAttribute("data-key")
      );

      const targetKey = currentElementMouseOn.getAttribute("data-key");
      // if (!targetKey) return;
      // nếu không có thì append vào cái cuối cùng
      // TODO: KIỂM TRA XEM CÓ BỊ TRÙNG KO 

      if (!targetKey) {
        if (mode === "same") {
          if (sourceContainer == "searchImages") return;
          const remainingImages = sourceContainerData.filter(img => !selectedKeys.includes(getKey(img)));
          const selectedImages = sourceContainerData.filter(img => selectedKeys.includes(getKey(img)));
          const newImages = [
            ...remainingImages,
            ...selectedImages
          ];
          updateQuestionField({ [sourceContainer]: newImages });
        }
        else if (mode === "different") {
          const remainingImages = sourceContainerData.filter(img => !selectedKeys.includes(getKey(img)));

          const selectedImages = sourceContainerData.filter(img => selectedKeys.includes(getKey(img)));
          const uniqueMap = new Map();
          [...toContainerData, ...selectedImages].forEach(img => {
            const imageKey = getKey(img);
            if (!uniqueMap.has(imageKey)) uniqueMap.set(imageKey, img);
          })
          const newImages = Array.from(uniqueMap.values());
          updateQuestionField({
            [toContainer]: newImages,
            [sourceContainer]: remainingImages
          })
        }
      }

      else {
        // TODO: BÌNH THƯỜNG MODE
        // vị trí drop 
        const dropIndex = toContainerData.findIndex(img =>
          getKey(img) === targetKey
        );

        if (mode === "same") {
          if (sourceContainer == "searchImages") return;
          const remainingImages = sourceContainerData.filter(img =>
            !selectedKeys.includes(getKey(img))
          );

          // Extract selected image objects
          const selectedImages = sourceContainerData.filter(img =>
            selectedKeys.includes(getKey(img))
          );

          // Insert selected at dropIndex
          const newImages = [
            ...remainingImages.slice(0, dropIndex),
            ...selectedImages,
            ...remainingImages.slice(dropIndex),
          ];

          updateQuestionField({ [sourceContainer]: newImages });
          // update 2 cái array
        } else if (mode === "different") {
          // Filter out selected images from original array

          const remainingImages = sourceContainerData.filter(img =>
            !selectedKeys.includes(getKey(img))
          );

          // Extract selected image objects
          const selectedImages = sourceContainerData.filter(img =>
            selectedKeys.includes(getKey(img))
          );

          const uniqueMap = new Map();
          [...toContainerData.slice(0, dropIndex), ...selectedImages, ...toContainerData.slice(dropIndex)].forEach(img => {
            const imageKey = getKey(img);
            if (!uniqueMap.has(imageKey)) uniqueMap.set(imageKey, img);
          })
          const newImages = Array.from(uniqueMap.values());

          updateQuestionField({
            [toContainer]: newImages,
            [sourceContainer]: remainingImages
          })
        }
      }

    }

    document.addEventListener("mouseup", handleDragUp);

    // might add later
    // const handleRightClick = (e) => {
    //   if (ref.current && !ref.current.contains(e.target)) return;
    //   const selectedElements = document.querySelectorAll("#selecto .image.selected");
    //   if (selectedElements.length > 0) return;

    //   const currentElementMouseOn = document.elementFromPoint(e.clientX, e.clientY);
    //   if (currentElementMouseOn && currentElementMouseOn.classList.contains("image")) {
    //     const selectedElements = document.querySelectorAll("#selecto .image.selected");
    //     selectedElements.forEach(el => el.classList.remove("selected"));
    //     currentElementMouseOn.classList.add("selected");
    //     selectoRef.current.setSelectedTargets([currentElementMouseOn]);
    //   }
    // }
    // document.addEventListener("mousedown", handleRightClick);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("mouseup", handleDragUp); 1
      // document.removeEventListener("mousedown", handleRightClick);
    };
  }, [searchImages, undoArray, redoArray]);

  return (
    <div className="relative flex flex-col elements w-full h-full searchImages">
      <Box className="sticky flex items-center h-fit">
        {/* <Button className="h-[56px]"
          disabled={undoArray?.length === 0}
          onClick={undo}>↩️</Button>
        <Button className="h-[56px]"
          disabled={redoArray?.length === 0}
          onClick={redo}>↪️</Button> */}
        <Select
          value={sortOption}
          label="Sort By"
          onChange={(e) => {
            const sortOption = e.target.value;
            setSortOption(sortOption);
            // sortImages(sortOption);
          }}
        >
          {/* <MenuItem value="d">Default</MenuItem> */}
          <MenuItem value="g">Name</MenuItem>
          <MenuItem value="hc">High Confidence</MenuItem>
        </Select>
        <Select
          value={groupOption}
          label="Group By"
          onChange={(e) => {
            const groupOption = e.target.value;
            setGroupOption(groupOption);
            // groupImages(groupOption);
          }}
        >
          {/* <MenuItem value="n">None</MenuItem> */}
          <MenuItem value="g">Group</MenuItem>
          <MenuItem value="hc">High Confidence</MenuItem>
        </Select>
      </Box>
      <div className="relative overflow-y-scroll flex-1 selecto-container" ref={ref} tabIndex={0}>
        <Selecto
          ref={selectoRef}

          dragContainer={".selecto-container"}
          selectableTargets={["#selecto .image"]}
          onSelect={e => {
            e.added.forEach(el => {
              el.classList.add("selected");
            });
            e.removed.forEach(el => {
              el.classList.remove("selected");
            });
          }}
          hitRate={0}
          selectByClick={true}
          selectFromInside={true}
          continueSelect={false}
          continueSelectWithoutDeselect={true}
          toggleContinueSelect={"shift"}
          toggleContinueSelectWithoutDeselect={[["ctrl"], ["meta"]]}
          ratio={0}
          // scrollOptions={{ container: ref.current, threshold: 30, speed: 15 }}
          innerScrollOptions={{ container: ref.current, threshold: 30, speed: 15 }} // might fix later
        ></Selecto>

        <div className={clsx("gap-4 p-4", { "grid": groupOption === "n" })} id="selecto">
          {/* {groupOption === "n" ? searchImages?.map((image) => {
            const src = getImage(blobs, getImageKey(image.keyframe_id, image.video_id, image.group_id));
            return (
              <figure className="relative image p-2 hover:bg-[rgba(68,171,255,0.15)] [&_*]:select-none [&_*]:pointer-events-none search"
                key={`${image.keyframe_id}-${image.video_id}-${image.group_id}`}
                data-key={`${image.key}-${image.video_id}-${image.group_id}`}
                data-container={"searchImages"}
                onDoubleClick={() => handleOpen(image)}
              >
                <img src={src}
                // onError={(e) => {
                //     e.target.src = ""
                // }}
                />
                <figcaption className="flex flex-row justify-between ">
                  <Typography variant="caption" className=" text-center text-black bg-opacity-50 p-1 rounded">
                    L{image.group_id} / V{image.video_id} / {image.keyframe_id}
                  </Typography>
                  <Typography className={clsx(image.score > 0.95 ? "text-blue-300" : image.score > 0.9 ? "text-yellow-500" : image.score > 0.8 ? "text-gray-400" : image.score > 0.7 ? "text-orange-900" : "")}>{image.score}</Typography>
                </figcaption>
              </figure>
            )
          })
            : */}
          {
            searchMode == "single" ?
              groupedSearchImages?.map((image) => {
                return (
                  <Disclosure key={image.group}>
                    <DisclosureButton className=" text-left px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded">
                      {image.group}
                    </DisclosureButton>
                    <DisclosurePanel className={clsx("grid p-4 max-h-[60vh] overflow-auto", cols)} id="selecto">
                      {
                        image.images.map((img) => {
                          const src = getImage(blobs, getKey(img));
                          return (
                            <figure className="relative image p-2 hover:bg-[rgba(68,171,255,0.15)] [&_*]:select-none [&_*]:pointer-events-none search"
                              key={getKey(img)}
                              data-key={getKey(img)}
                              data-container={"searchImages"}
                              onDoubleClick={() => handleOpen(img)}
                            >
                              <img src={src}
                              // onError={(e) => {
                              //     e.target.src = ""
                              // }}
                              />
                              <figcaption className="flex flex-row justify-between ">
                                <Typography variant="caption" className=" text-center text-black bg-opacity-50 p-1 rounded">
                                  {img.group_id} / {img.video_id} / {img.keyframe_id}
                                </Typography>
                                <Typography className={clsx(img.score > 0.95 ? "text-blue-300" : img.score > 0.9 ? "text-yellow-500" : img.score > 0.8 ? "text-gray-400" : img.score > 0.7 ? "text-orange-900" : "")}>{img.score.toFixed(4)}</Typography>
                              </figcaption>
                            </figure>
                          )
                        })
                      }
                    </DisclosurePanel>
                  </Disclosure>
                )
              })
              :
              groupedSearchImagesChain?.map((imageGroup) => {
                return (
                  <Disclosure key={imageGroup.group}>
                    <DisclosureButton className=" text-left px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded">
                      {imageGroup.group}
                    </DisclosureButton>
                    <DisclosurePanel className=" p-4 overflow-auto space-y-2" id="selecto">
                      {
                        imageGroup.images.map((imageChain, index) => {
                          return (
                            <Paper className="flex flex-row justify-between image search hover:!bg-[rgba(68,171,255,0.15)]"
                              key={getKey(imageChain)}
                              data-key={getKey(imageChain)}
                              data-container={"searchImages"}
                            >
                              {
                                imageChain.map((image) => {
                                  return (
                                    <figure className="relative p-2 [&_*]:select-none [&_*]:pointer-events-none"
                                      key={getImageKey_(image)}
                                      onDoubleClick={() => handleOpen(image)}
                                    >
                                      <img src={imagePath(getImageKey_(image))}
                                      />
                                      <figcaption className="flex flex-row justify-between ">
                                        <Typography variant="caption" className=" text-center text-black bg-opacity-50 p-1 rounded">
                                          {image.group_id} / {image.video_id} / {image.keyframe_id}
                                        </Typography>
                                        <Typography className={clsx(image.score > 0.95 ? "text-blue-300" : image.score > 0.9 ? "text-yellow-500" : image.score > 0.8 ? "text-gray-400" : image.score > 0.7 ? "text-orange-900" : "")}>{image.score.toFixed(4)}</Typography>
                                      </figcaption>
                                    </figure>
                                  )
                                })
                              }
                            </Paper>)
                        })
                      }
                    </DisclosurePanel>
                  </Disclosure>
                )
              })
          }
        </div>
      </div>
    </div>
  )
}

const rank = {
  "Very High": 5,
  "High": 4,
  "Good": 3,
  "Average": 2,
  "Low": 1,
};
